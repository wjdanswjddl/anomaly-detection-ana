# Managing VAE/CAE Anomaly-Detector Training Runs

This document is written so that a future agent (or person) can launch,
monitor, resume, and validate a training run of the VAE or CAE anomaly
detectors without re-deriving anything. Architectures and citations:
`guided_diffusion/autoencoder.py` and `CITATIONS.md`.

## What these models are

Reconstruction-based anomaly detectors, trained on nominal images only.
Inference passes an image through the bottleneck and takes
`saliency = input - reconstruction` as the anomaly map (same convention as
the diffusion round-trip in the comparison notebooks: plotted with
`cmap="bwr", vmin=-0.1, vmax=0.1`).

- **VAE** (`--ae_type vae`): conv VAE, loss = pixel MSE + `kld_weight` × KLD.
- **CAE** (`--ae_type cae`): conv autoencoder, loss = pixel MSE (`l2`), with
  optional SSIM term (`--ssim_weight > 0`).

## Environment

- The checked-in `env/` virtualenv is **broken on this host** (it symlinks a
  `/opt/conda` that no longer exists). Use any python with `torch`, `h5py`,
  `blobfile`, `scipy`, `matplotlib`, `tqdm` (e.g. the system `python3`).
- Check GPU availability first:
  `python3 -c "import torch; print(torch.cuda.is_available())"`.
  CPU works for smoke tests (minutes for a few steps at 512×512); real
  training needs a GPU-enabled torch.
- Run everything from the repo root
  (`.../Diffusion-Anomaly-Detection/diffusion-anomaly`).

## Data

| Experiment | Directory | Format |
|---|---|---|
| SBND | `/scratch/7DayLifetime/gputnam/raw-sq-wtruth/` | `.npz` with `reco`/`truth` `(N,1,512,512)` |
| ICARUS | `/exp/sbnd/data/users/gputnam/DNN-ROI-images/` | `.h5`, `event_N/deconvolved_signal` (+ `true_number_electrons`), tiled to 512×512 by the loader |

**Warning:** the SBND directory lives on `/scratch/7DayLifetime/` — a
purge-policy scratch area. Verify it still exists before launching
(`ls` a few files); if purged, ask the user for a fresh dataset location.
ICARUS `deconvolved_signal` images are divided by
`DECONV_SIGNAL_SCALE = 2.0` (`guided_diffusion/image_datasets.py`) before
the [-1, 1] clip.

## Launching training

```bash
source model_flags_VAE_SBND.sh    # or CAE_SBND / VAE_ICARUS / CAE_ICARUS
python3 scripts/autoencoder_train.py $AE_TRAIN_FLAGS --log_dir ./results/vae-sbnd
```

Each flag script exports `$MODEL_FLAGS`, `$TRAIN_FLAGS`, `$DATADIR`, and the
combined `$AE_TRAIN_FLAGS`. Anything can be overridden by appending flags
(later flags win): e.g. `--batch_size 4 --microbatch 2` on a small GPU.

Key flags (defaults in `script_util.autoencoder_defaults()` and
`scripts/autoencoder_train.py`):

- `--ae_type vae|cae`
- `--ae_hidden_dims 32,64,128,256,512,512,512` — one stride-2 encoder stage
  per entry; `image_size` must be divisible by `2**len(hidden_dims)`.
- `--ae_latent_dim 512`; `--kld_weight 1e-4` (VAE); `--spatial_latent`,
  `--ssim_weight` (CAE).
- `--lr`, `--batch_size`, `--microbatch` (must be ≥ 2 or -1: BatchNorm cannot
  train on batches of 1), `--save_interval`, `--plot_interval`,
  `--validation_interval`, `--log_dir`.
- `--use_fp16 False` — keep off; fp16 is untested for the autoencoders.

Smoke-test hook: `DIFFUSION_TRAINING_TEST=1` makes the loop exit right after
its first checkpoint save, e.g.:

```bash
DIFFUSION_TRAINING_TEST=1 python3 scripts/autoencoder_train.py $AE_TRAIN_FLAGS \
    --batch_size 2 --save_interval 5 --validation_interval 5 \
    --plot_interval 1000000 --log_dir ./results/vae-smoke
```

## Monitoring

Everything lands in `--log_dir`:

- `progress.csv` / `log.txt` — per-`log_interval` metrics. Columns: `loss`,
  and `mse`/`kld` (VAE) or `l2`(/`ssim`) (CAE); `*_q0` are the same values
  (there is only one "timestep" bucket); `val-*_q0` are the validation-batch
  variants. Note: the *unprefixed* `loss`/`mse`/... columns mix train and
  validation batches (pre-existing logger behavior) — use `val-*_q0` vs
  `*_q0` to separate them. Also `step`, `samples`, `grad_norm`, `param_norm`.
- `tb/` — TensorBoard events (`tensorboard --logdir <log_dir>/tb`).
- `validation-plots/step-N-ae/img-K/` — every `--plot_interval` steps:
  `img.png` (original), `reco.png`, `residual.png` (reconstruction −
  original, bwr ±0.1), `reco_wavf_I*.png` waveform overlays.

Healthy behavior:

- **VAE**: `mse` falls from O(1e-1) toward O(1e-3) (LArTPC images are mostly
  empty); `kld` rises from ~0 then plateaus. If `kld` collapses to ~0 while
  `mse` stays flat (posterior collapse), lower `--kld_weight`; if
  reconstructions are blurry/washed out, raise it cautiously or lower `lr`.
- **CAE**: `l2` decreases monotonically (modulo noise).
- **Both**: the failure mode to watch is the model trivially learning the
  near-empty background. Check `validation-plots/`: reconstructions must
  show actual tracks, not uniform noise/blankness, and residuals should not
  simply reproduce every track.

## Checkpoints and resuming

Saved to `--log_dir` every `--save_interval` steps:
`model{step:06d}.pt`, `ema_{rate}_{step:06d}.pt` (EMA weights, default rate
0.9999), `opt{step:06d}.pt` (optimizer state).

Resume with:

```bash
python3 scripts/autoencoder_train.py $AE_TRAIN_FLAGS \
    --log_dir ./results/vae-sbnd \
    --resume_checkpoint ./results/vae-sbnd/model010000.pt
```

The step is parsed from the filename; matching `ema_*`/`opt*` files are
auto-loaded from the same directory. For inference after long runs prefer
the `ema_*` checkpoint (load it as `--model_path`); at smoke scale EMA is
meaningless — use `model*.pt`.

## Inference / anomaly maps

```bash
source model_flags_VAE_SBND.sh
python3 scripts/autoencoder_sample.py $MODEL_FLAGS \
    --model_path ./results/vae-sbnd/model010000.pt \
    --data_dir /scratch/7DayLifetime/gputnam/raw-sq-wtruth/ \
    --batch_size 4 --num_batches 10 --output_dir ./results/vae-sbnd/ae_samples
```

Outputs per batch: `batch_%04d.npz` with `input`, `reco`,
`saliency = input - reco` (all float32 `(N,1,H,W)`) and `paths`; plus
3-panel PNGs per image unless `--save_png False`. The `.npz` contents slot
directly into the `notebooks/CompareModels*` comparison convention, and any
loaded model exposes `model.reconstruct(imgs)` / `model.anomaly_map(imgs)`
for notebook use alongside the diffusion round-trip.

**The model flags at inference must match training** (`ae_type`,
`image_size`, `ae_hidden_dims`, `ae_latent_dim`, `spatial_latent`) or the
state dict will not load — sourcing the same flag script is the easy way.

## Validation checklist (run before a long job)

1. `python3 -m pytest tests/ -v` — all tests pass.
2. Smoke train each config you intend to run (command above with
   `DIFFUSION_TRAINING_TEST=1`); confirm `model000005.pt` appears and
   `progress.csv` has finite losses.
3. Smoke inference on the resulting checkpoint; open one PNG and confirm the
   three panels render.
4. Confirm the data directory is still readable and non-empty.

## Trained SBND models (September 2026)

Both were trained on the SBND raw h5 files streamed over xrootd (file lists
in each run directory's `filelists/`; flags in `model_flags_{VAE,CAE}_SBND.sh`
plus `--batch_size 64 --microbatch -1` for the VAE).

| Model | Run directory | Checkpoint to use |
|---|---|---|
| VAE | `/scratch/7DayLifetime/gputnam/training-SBND-VAE/iterA` | `ema_0.9999_263000.pt` (training stopped at step 263000 on 2026-09-04) |
| CAE | `/scratch/7DayLifetime/gputnam/training-SBND-CAE/iterA` | `ema_0.9999_612000.pt` (or the latest `ema_*` if training has continued) |

**Warning:** both run directories are on `/scratch/7DayLifetime/`, which is
purged after seven days. Copy the checkpoints you want to keep somewhere
durable and pass them with `--model_path`.

### Minimal inference demos

`scripts/vae_sbnd_demo.py` and `scripts/cae_sbnd_demo.py` hard-code the
trained configuration and default to the checkpoints above. Each script

1. generates `--num_samples` images from random noise (`model.sample`; for
   the VAE this decodes `z ~ N(0, I)`, for the CAE it decodes random latent
   codes scaled by `--latent_scale` - the CAE has no prior, so treat this as
   a decoder probe), writing `generated.npz` / `generated.png`;
2. loads one SBND file (`--input`, default: a validation file in dCache),
   runs anomaly detection on the `--num_tiles` tiles with the most signal,
   and writes `anomaly.npz` (`input`, `reco`, `saliency`, `tile_index`) plus
   a 3-panel `tile_XX.png` per tile, printing a mean-squared-residual score.

```bash
# from the repo root; needs a valid SBND bearer token
# (htgettoken -a htvaultprod.fnal.gov -i sbnd) for the default root:// input
LD_PRELOAD=/usr/lib64/libXrdPosixPreload.so HDF5_USE_FILE_LOCKING=FALSE \
    python3 scripts/vae_sbnd_demo.py --output_dir ./results/vae-sbnd-demo
LD_PRELOAD=/usr/lib64/libXrdPosixPreload.so HDF5_USE_FILE_LOCKING=FALSE \
    python3 scripts/cae_sbnd_demo.py --output_dir ./results/cae-sbnd-demo

# local input (no preload needed): any SBND raw .h5 or reco/truth .npz
python3 scripts/vae_sbnd_demo.py --input /path/to/file.h5 --model_path /path/to/ema.pt
```

`guided_diffusion.image_datasets.load_image_file(path, 512)` is the
single-file loader the demos use; it returns the same `[-1, 1]` tiles the
training loader feeds the models, for use in notebooks.
