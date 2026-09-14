import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger, validation_plots
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from visdom import Visdom
from tqdm.auto import tqdm
# viz = Visdom(port=8850)
viz = Visdom(port=8850, server="sbndbuild03.fnal.gov")
import numpy as np

INITIAL_LOG_LOSS_SCALE = 20.0

def visualize(img):
    return np.clip(img, -1, 1)
    _min = img.min(axis=(1, 2, 3))
    _max = img.max(axis=(1, 2, 3))
    normalized_img = (img.T - _min)/ (_max - _min)
    return normalized_img.T

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        validation,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        validation_interval,
        save_interval,
        plot_interval,
        resume_checkpoint,
        use_fp16=False,
        weight_batches=False,
        weight_pixels=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.datal = data
        self.validationl = validation
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.validation_interval = validation_interval
        self.save_interval = save_interval
        self.plot_interval = plot_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.weight_batches = weight_batches
        self.weight_pixels = weight_pixels

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            print('resume model')
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        i = 0

        pbar = tqdm()
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.datal)
            cond.pop("path", None)
            bw = cond.pop("weight", None)
            pw = cond.pop("pixel_weight", None)
            batch_weights = None if not self.weight_batches else bw
            pixel_weights = None if not self.weight_pixels else pw

            self.run_step(batch, cond, batch_weights=batch_weights, pixel_weights=pixel_weights)

            if self.step % self.validation_interval == 0:
                vbatch, vcond = next(self.validationl)
                vcond.pop("path", None)
                bw = vcond.pop("weight", None)
                pw = vcond.pop("pixel_weight", None)
                vbatch_weights = None if not self.weight_batches else bw
                vpixel_weights = None if not self.weight_pixels else pw
                self.run_validation_step(vbatch, vcond, batch_weights=vbatch_weights, pixel_weights=vpixel_weights)

            if self.step % self.log_interval == 0:
                logger.dumpkvs()

            if self.step % self.save_interval == 0 and self.step > 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return

            if self.step % self.plot_interval == 0 and self.step > 0:
                # make validation plots
                vbatch, _ = next(self.validationl)
                for ibatch in range(min(4, vbatch.shape[0])):
                    outdir = logger.Logger.CURRENT.dir + "/validation-plots/step-%i-ddpm/img-%i/" % (self.step, ibatch)
                    os.makedirs(outdir, exist_ok=True)
                    with th.no_grad():
                        validation_plots.validation_plots(self.diffusion, self.ddp_model, outdir, vbatch[ibatch])
                for ibatch in range(min(4, vbatch.shape[0])):
                    outdir = logger.Logger.CURRENT.dir + "/validation-plots/step-%i-ddim/img-%i/" % (self.step, ibatch)
                    os.makedirs(outdir, exist_ok=True)
                    with th.no_grad():
                        validation_plots.validation_plots(self.diffusion, self.ddp_model, outdir, vbatch[ibatch], ddpm=False)

            self.step += 1
            pbar.update(1)

        pbar.close()
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_validation_step(self, batch, cond, batch_weights=None, pixel_weights=None):
        with th.no_grad():
            self.forward_backward(batch, cond, validation=True, batch_weights=batch_weights, pixel_weights=pixel_weights)

    def run_step(self, batch, cond, batch_weights=None, pixel_weights=None):
        self.forward_backward(batch, cond, batch_weights=batch_weights, pixel_weights=pixel_weights)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond, validation=False, batch_weights=None, pixel_weights=None):
        self.mp_trainer.zero_grad()
        all_losses = {}
        ts = th.empty(0)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
       
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            if batch_weights is not None: 
                weights = weights * batch_weights[i : i + self.microbatch].to(dist_util.dev())
            pweights = None if pixel_weights is None else pixel_weights[i : i + self.microbatch].to(dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                pixel_wgt=pweights,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses1 = compute_losses()
                losses = losses1[0]
                loss = (losses["loss"] * weights).mean()
                if not validation:
                    self.mp_trainer.backward(loss)
            else:
                with self.ddp_model.no_sync():
                    losses1 = compute_losses()
                    losses = losses1[0]
                    loss = (losses["loss"] * weights).mean()
                    if not validation:
                        self.mp_trainer.backward(loss)

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            # Keep track of the losses for logging
            for k, v in losses.items():
                w_v = weights*v
                if k in all_losses:
                    all_losses[k] = th.cat([all_losses[k], w_v.detach()], dim=0)
                else:
                    all_losses[k] = w_v.detach()
            ts = th.cat([ts, t.detach().cpu()])


        log_loss_dict(
            self.diffusion, ts, all_losses,
            validation=validation
        )

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"brats2update{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"emabrats2update_{rate}_{(self.step+self.resume_step):06d}.pt"
                print('filename', filename)
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"optbrats2update{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier(device_ids=[0])


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses, validation=False):
    prefix = "val-" if validation else ""
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{prefix}{key}_q{quartile}", sub_loss)
