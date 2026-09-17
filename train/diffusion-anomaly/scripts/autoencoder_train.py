"""
Train a VAE or CAE anomaly detector on images.

Mirrors image_train.py, but builds an autoencoder (guided_diffusion/
autoencoder.py) instead of a diffusion model. Select the model with
--ae_type vae|cae. See TRAINING_AUTOENCODERS.md and CITATIONS.md.
"""
import sys
import argparse
sys.path.append("..")
sys.path.append(".")
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.autoencoder import AEObjective
from guided_diffusion.resample import UniformSampler
from guided_diffusion.script_util import (
    autoencoder_defaults,
    create_autoencoder,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import AETrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)

    logger.log("creating autoencoder (%s)..." % args.ae_type)
    model = create_autoencoder(
        **args_to_dict(args, autoencoder_defaults().keys())
    )
    model.to(dist_util.dev())
    objective = AEObjective()
    # The autoencoder has no diffusion timesteps; this sampler always yields
    # t=0 with unit weights.
    schedule_sampler = UniformSampler(objective, maxt=1)

    logger.log("creating data loader...")
    datal = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            charge_scale=args.charge_scale,
            class_cond=False,
    )

    validationl = load_data(
            data_dir=args.validation_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            charge_scale=args.charge_scale,
            class_cond=False,
            require_charge=True,
    )

    logger.log("training...")
    AETrainLoop(
        model=model,
        diffusion=objective,
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


def create_argparser():
    defaults = dict(
        data_dir="",
        validation_dir="",
        log_dir="./results",
        charge_scale=1.,
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=12,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=50,
        validation_interval=5,
        plot_interval=1200,
        save_interval=400,
        resume_checkpoint='',
        use_fp16=False,
        fp16_scale_growth=1e-3,
        weight_batches=False,
        weight_pixels=False,
        max_steps=0,  # 0 = run until interrupted; set e.g. 263000 to match Gray VAE
    )
    defaults.update(autoencoder_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
