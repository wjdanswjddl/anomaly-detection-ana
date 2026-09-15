# Anomaly detection analysis (SBND / ICARUS)

Workflow notebooks and scripts for LArTPC diffusion-based anomaly detection.
Training / GPU inference code is [gputnam/diffusion-anomaly](https://github.com/gputnam/diffusion-anomaly);
run those steps on **EAF** via the `conda env:.conda-diffusion` Jupyter kernel.

## Layout

| Path | Role |
|------|------|
| `configs/paths.py` | Unified APP / DATA / SCRATCH roots and naming |
| `samples/` | wvfm→h5, inspect, handscan, synthetic defects |
| `train/` | EAF probe + diffusion/classifier training + learning curves |
| `inference/` | DDIM / mixed reconstruction runners |
| `analysis/` | Score distributions, ROC, batch `analyze_outputs.py` |
| `archive/` | Superseded notebooks (see `archive/README.md`) |
| `_stubs/` | Visdom stub for CPU hosts |

**Code (this repo):** `/exp/sbnd/app/users/munjung/anomaly-detection`  
**Outputs / data:** `/exp/sbnd/data/users/munjung/anomaly-detection`  
**EAF scratch (GPU):** `/scratch/7DayLifetime/munjung/anomaly-detection` (plus legacy `.../ICARUS`)

Compatibility symlinks at the repo root keep old import paths working
(`run_ddim2ddim_inference.py`, `handscan_validation`, …).

## Workflow

### 1. Samples (local `env` kernel)

1. `samples/01_WvfmToH5.ipynb` — or `samples/grid_wvfm2h5/` for production
2. `samples/02_InspectAndHandscan.ipynb` → `HandscanUnhealthy.ipynb`
3. `samples/h5_to_npz.ipynb` — h5 → training/inference NPZ
4. `samples/03_InsertDefects.ipynb` → `apply_detector_defects_npz.py`

### 2. Train (EAF `conda-diffusion` kernel)

1. **`train/00_ExploreRemoteEAF.ipynb`** — map remote code, scratch, checkpoints; import EAF flag scripts  
2. **Two GPU slices:** `train/01a_TrainDiffusion_sliceA.ipynb` (`linear`→`ramp`→`pred_xstart`) and `train/01b_TrainDiffusion_sliceB.ipynb` (`anisotropic`→`cosine`); or single-GPU `train/01_TrainDiffusion.ipynb` for all five sequentially  
3. `train/02_TrainClassifier.ipynb`  
4. `train/03_LearningCurves.ipynb`

### 3. Inference + metrics (EAF kernel for scratch I/O)

1. `inference/01_RunInference.ipynb`  
2. `inference/02_CompareReconstructions.ipynb`  
3. `analysis/01_MetricsAndROC.ipynb` or `python analysis/analyze_outputs.py`  
4. `analysis/CompareROCCurves.ipynb` — overlay saved ROC JSON

## Naming

Inference pickles: `{stem}_T{T}_{mode}.pkl` with
`mode ∈ {ddim2ddim, rand2ddim, rand2ddpm, ddpm2ddpm, ddim2ddpm}`.

Defect tags: `bad_wire`, `coh_noise`, `charge_tail`.  
Model suffix in output dirs: `""`, `-ramp`, `-anisotropic`, `-cosine`.

Helpers: `configs.paths.resolve_inference_dirs`, `inference_pickle_name`.

## Quick commands

```bash
# Defect injection
python samples/apply_detector_defects_npz.py --input-dir /path/to/healthy_npz

# DDIM→DDIM inference (from repo root)
PYTHONPATH=_stubs:train/diffusion-anomaly:inference \
  python inference/run_ddim2ddim_inference.py \
    --input-dir ... --output-dir ... --model-path ... --T 200

# Batch ROC / figures → data area
python analysis/analyze_outputs.py
```
