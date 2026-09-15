# Archive map

Moved here during the 2026-09 layout cleanup. Prefer the stage notebooks under
`samples/`, `train/`, `inference/`, `analysis/`.

## Notebooks

| Archived | Replaced / continued by |
|----------|-------------------------|
| `wvfm2h5.ipynb` | `samples/01_WvfmToH5.ipynb`, `samples/wvfm2h5.py`, `samples/grid_wvfm2h5/` |
| `inspect_samples-ICARUS.ipynb` | `samples/02_InspectAndHandscan.ipynb`, `samples/inspect_samples.ipynb` |
| `make_diseased_samples.ipynb` | `samples/03_InsertDefects.ipynb`, `samples/apply_detector_defects_npz.py` |
| `PrepareHandscanValidation.ipynb` | Still useful; open from here. Entry: `samples/02_InspectAndHandscan.ipynb` |
| `InferHandscanValidation.ipynb` | `inference/run_handscan_validation_inference.py`, `inference/01_RunInference.ipynb` |
| `AnalyzeHandscanValidation.ipynb` | `analysis/01_MetricsAndROC.ipynb` |
| `RunDDIM2DDIM_SingleExample.ipynb` | `inference/02_CompareReconstructions.ipynb` |
| `CompareReconstructions-Stitched.ipynb` | `inference/CompareReconstructions-ICARUS.ipynb` |
| `ListCompareReconstructions.ipynb` | same |
| `learning_curve.ipynb` | `train/03_LearningCurves.ipynb` |
| `translation.ipynb` | `train/01_*` / `inference/01_RunInference.ipynb` |
| `detect_anomaly.ipynb`, `classifier_validation.ipynb` | obsolete prototypes |
| `data_jobs.ipynb`, `reco_jobs.ipynb`, `diffusion-anomaly.ipynb` | superseded by CLI + stage notebooks |

## Data products

Large / old products were moved to the parallel data tree:

`/exp/sbnd/data/users/munjung/anomaly-detection/archive/`

| Former app path | New location |
|-----------------|--------------|
| `validation/`, `validation_file18/`, `validation_noclass/` | `.../archive/validation*` |
| `translation/` | `.../archive/translation` |
| `handscan_archives/` | `.../archive/handscan_archives` |
| `handscan_validation/` | `.../samples/handscan/validation` (symlink `handscan_validation` at repo root) |
| `logs/`, `figures_paper/`, `code_submission/` | `.../archive/` |
| root `file*.h5` | `.../samples/h5/` |
| `ICARUS_NNs/` | `.../archive/ICARUS_NNs` |
