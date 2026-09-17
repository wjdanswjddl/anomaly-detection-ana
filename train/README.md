# Training (EAF GPU)

## Five diffusion configs

| name | flags | notes |
|------|-------|-------|
| `linear` | `configs/train_flags/linear.sh` | nominal / Ho linear β (**Gray iterE schedule**) |
| `cosine` | `configs/train_flags/cosine.sh` | cosine β |
| `ramp` | `configs/train_flags/ramp.sh` | piecewise ramp β |
| `anisotropic` | `configs/train_flags/anisotropic.sh` | linear β + spatial noise |
| `pred_xstart` | `configs/train_flags/pred_xstart.sh` | predict \(x_0\) |

`train/diffusion-anomaly` includes `ramp` schedule + `anisotropic_diffusion.py` (synced from the code_submission tree).

Shared defaults in `configs/train_flags/_common.sh` match
`/exp/sbnd/data/users/gputnam/training-SBND/iterE/flags.txt`
(batch 8 / microbatch 4 for MIG; iterE used 50/10 — raise via BATCH_SIZE if headroom)

plus wall-time fixes: `validation_interval=500`, `plot_interval=10000`, `use_fp16=True`, `max_steps=111000`.

## Steps

1. `00_ExploreRemoteEAF.ipynb` — map EAF paths; import any remote `*flags*.sh` into `configs/train_flags/imported/`
2. **Two GPU slices:** open `01a_TrainDiffusion_sliceA.ipynb` (`linear`→`ramp`→`pred_xstart`) and `01b_TrainDiffusion_sliceB.ipynb` (`anisotropic`→`cosine`) on separate kernels; set `DRY_RUN = False` in each.
   Single GPU: `01_TrainDiffusion.ipynb` with `SEQUENTIAL=True`
3. `02_TrainClassifier.ipynb` — classifier (optional / separate)
4. `03_LearningCurves.ipynb` — point `RUN_DIR` at `.../training/diffusion/<config>/` (**SBND**)  
5. `03c_InspectICARUS_AE.ipynb` — VAE/CAE under `.../training/{vae,cae}/icarus/`

Prefer the **EMA** checkpoint near step 111000 for inference (same convention as Gray’s iterE).

Logs default to `/scratch/<existing-7Day-pool>/munjung/anomaly-detection/training/diffusion/<config>/`.

## VAE / CAE baselines

| Notebook | Purpose |
|----------|---------|
| `04a_TrainVAE_ICARUS.ipynb` | Train VAE on DNN-ROI (GPU slice A) |
| `04b_TrainCAE_ICARUS.ipynb` | Train CAE on DNN-ROI (GPU slice B) |
| `stop_running_trainers.sh` | Kill leftover `image_train` / AE trainers on EAF |
| Flags | `configs/train_flags/{vae,cae}_{sbnd,icarus}.sh` |
| Library | `train/diffusion-anomaly/guided_diffusion/autoencoder.py` |
| Docs | `train/diffusion-anomaly/TRAINING_AUTOENCODERS.md` |

SBND AE weights: copy Gray’s EMA into
`/exp/sbnd/data/users/munjung/anomaly-detection/training/{vae,cae}/sbnd/`
(scratch `training-SBND-{VAE,CAE}/iterA` is often purged).

Before starting AE training, free the GPUs:

```bash
# on EAF jupyter-munjung
bash train/stop_running_trainers.sh
rm -f STOP_TRAINING
```
