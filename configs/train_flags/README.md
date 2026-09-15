# Diffusion training flag bundles (5 configs)

| Script | Tag | Key flags |
|--------|-----|-----------|
| `linear.sh` | `linear` / nominal | `noise_schedule=linear`, ε-pred (**matches Gray iterE schedule**) |
| `cosine.sh` | `cosine` | `noise_schedule=cosine` |
| `ramp.sh` | `ramp` | `noise_schedule=ramp` |
| `anisotropic.sh` | `anisotropic` | `anisotropic_noise=True` (linear β) |
| `pred_xstart.sh` | `pred_xstart` | `predict_xstart=True` (linear β) |

Shared architecture / optimizer: `_common.sh`, aligned with
`/exp/sbnd/data/users/gputnam/training-SBND/iterE/flags.txt`:

| flag | value |
|------|-------|
| `batch_size` / `microbatch` | **8** / **4** (MIG-safe; iterE used 50/10 — override `BATCH_SIZE`) |
| `lr` / `weight_decay` | 1e-4 / 0.01 |
| `weight_batches` / `weight_pixels` | False |
| `save_interval` | 1000 |
| `charge_scale` | 1 (iterE omitted; argparse default) |
| architecture | 512, ch=32, mult=1,2,4,8,8,8, …

**Improvements vs bare iterE** (also in `_common.sh`):

| flag | value | why |
|------|-------|-----|
| `validation_interval` | 500 | code default was 5 |
| `plot_interval` | 10000 | iterE used 2000; plots are expensive |
| `use_fp16` | True | A100/MIG throughput |
| `max_steps` | 111000 | stop near iterE’s production ckpt (constant LR) |

```bash
export DATA_DIR=/scratch/7DayExclusive/munjung/anomaly-detection/npz
source configs/train_flags/cosine.sh
cd train/diffusion-anomaly
export OPENAI_LOGDIR=/scratch/.../training/diffusion/cosine
python scripts/image_train.py $IMAGE_TRAIN_FLAGS
```

`imported/` may hold flag scripts copied from EAF by `train/00_ExploreRemoteEAF.ipynb`.
