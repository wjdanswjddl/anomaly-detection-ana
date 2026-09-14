# Diffusion Models for LArTPC Anomaly Detection (SBND / ICARUS)

Anomaly detection on Liquid Argon Time Projection Chamber (LArTPC) detector
images from the SBND and ICARUS experiments at Fermilab. A denoising
diffusion probabilistic model (DDPM) is trained on nominal reconstructions;
classifier-guided sampling translates anomalous events toward the nominal
manifold; the per-pixel difference between the input and the reconstruction
is the anomaly map.

Forked from
[openai/guided-diffusion](https://github.com/openai/guided-diffusion) via
the MICCAI 2022 medical-imaging adaptation.

## Data format

The dataset loader (`guided_diffusion/image_datasets.py`) reads:

- **`.npz`** files with two arrays:
  - `reco` — reconstructed detector image, shape `(N, 1, H, W)`.
  - `truth` — true charge per pixel, same shape; used for importance
    sampling and pixel weighting.
- **`.h5`** files with raw waveform planes; charge is normalized per
  plane and tiled into `512×512` patches.

`reco` is normalized (min-max) per file at load time.

## Workflows

The three CLI entry points each take dozens of flags. The
`model_flags_SBND.sh` / `model_flags_SBND_anisotropic.sh` /
`classifier_flags.sh` shell scripts at the repo root export the
production flag bundles; source one and reference its variables on the
command line.

### 1. Train the diffusion model

```bash
source model_flags_SBND.sh
python scripts/image_train.py $IMAGE_TRAIN_FLAGS
```

Anisotropic noise (signal-proportional perturbations on the LArTPC
charge image) is enabled by `model_flags_SBND_anisotropic.sh` instead.

### 2. Train the classifier (on noisy diffusion-corrupted images)

```bash
source classifier_flags.sh
python scripts/classifier_train.py $CLASSIFIER_TRAIN_FLAGS
```

### 3. Run anomaly detection

```bash
python scripts/classifier_sample_known.py \
    --data_dir <path_to_test_npz> \
    --model_path ./results/model<step>.pt \
    --classifier_path ./results/classifier_model_<step>.pt \
    --classifier_scale 100 --noise_level 500 \
    $MODEL_FLAGS $DIFFUSION_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS
```

`run_translation.sh` is a working example.

## Checkpoints

Saved to the run's log directory as `model<step>.pt`,
`ema_<rate>_<step>.pt`, `opt<step>.pt` (diffusion training) and
`classifier_model_<step>.pt` / `classifier_opt_<step>.pt` (classifier
training).

## Tests

```bash
pytest tests/
```

Covers the diffusion math (`linear`/`cosine` schedules, `q_sample`,
respacing), the schedule samplers, the SBND `.npz` dataset path, the
UNet and classifier factory shapes, and the EMA / timestep-embedding
helpers.

## Distributed training

Uses `torch.distributed` via `guided_diffusion/dist_util.py`. Launch
multi-GPU runs with `mpiexec` or `torchrun`; single-GPU works without
any wrapper.
