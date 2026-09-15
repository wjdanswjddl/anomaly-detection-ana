"""
Train a diffusion model on images.
"""
import os
import sys
import argparse
import torch as th
import torch.distributed as dist

# Prefer non-NVML allocator path on MIG (set before CUDA context if possible).
os.environ.setdefault(
    "PYTORCH_ALLOC_CONF",
    os.environ.get(
        "PYTORCH_ALLOC_CONF",
        os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False"),
    ),
)

sys.path.append("..")
sys.path.append(".")
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def _log_cuda_mem(tag: str) -> None:
    if not th.cuda.is_available():
        logger.log(f"{tag}: CUDA unavailable")
        return
    # Avoid th.cuda.mem_get_info() — it uses NVML and can assert on MIG.
    logger.log(
        f"{tag}: device={th.cuda.get_device_name(0)} "
        f"alloc={th.cuda.memory_allocated()/1024**3:.2f}GiB "
        f"reserved={th.cuda.memory_reserved()/1024**3:.2f}GiB "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
        f"PYTORCH_ALLOC_CONF={os.environ.get('PYTORCH_ALLOC_CONF', '<unset>')}"
    )


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    log_dir = os.environ.get("OPENAI_LOGDIR")
    logger.configure(dir=log_dir)
    logger.log(f"OPENAI_LOGDIR={log_dir!r} -> logger.dir={logger.get_dir()}")
    _log_cuda_mem("startup")

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=1000)
    _log_cuda_mem("after_model")

    logger.log(
        f"batch_size={args.batch_size} microbatch={args.microbatch} "
        f"use_fp16={args.use_fp16} max_steps={args.max_steps}"
    )
    logger.log("creating data loader...")
    datal = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            charge_scale=args.charge_scale,
            class_cond=False,
            deterministic=False,
            num_workers=args.num_workers,
    )

    validationl = load_data(
            data_dir=args.validation_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            charge_scale=args.charge_scale,
            class_cond=False,
            require_charge=True,
            deterministic=True,
            num_workers=0,
    )

    logger.log("training...")
    try:
        TrainLoop(
            model=model,
            diffusion=diffusion,
            data=datal,
            validation=validationl,
            batch_size=args.batch_size,
            microbatch=args.microbatch,
            lr=args.lr,
            ema_rate=args.ema_rate,
            log_interval=args.log_interval,
            validation_interval=args.validation_interval,
            plot_interval=args.plot_interval,
            save_interval=args.save_interval,
            resume_checkpoint=args.resume_checkpoint,
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.lr_anneal_steps,
            max_steps=args.max_steps,
            weight_batches=args.weight_batches,
            weight_pixels=args.weight_pixels,
        ).run_loop()
    except RuntimeError as exc:
        _log_cuda_mem("on_error")
        raise


def create_argparser():
    # Defaults lean toward gputnam iterE architecture + MIG-safe batch.
    # Flag bundles in configs/train_flags/ override these at the CLI.
    defaults = dict(
        data_dir="",
        validation_dir="",
        charge_scale=1.,
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.01,
        lr_anneal_steps=0,
        max_steps=111000,
        batch_size=8,
        microbatch=4,
        ema_rate="0.9999",
        log_interval=50,
        validation_interval=500,
        plot_interval=10000,
        save_interval=1000,
        resume_checkpoint='',
        use_fp16=True,
        fp16_scale_growth=1e-3,
        weight_batches=False,
        weight_pixels=False,
        num_workers=0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
