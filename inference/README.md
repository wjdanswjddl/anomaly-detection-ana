# Inference / translation

| File | Purpose |
|------|---------|
| `01_RunInference.ipynb` | EAF launcher for DDIM / mixed modes |
| `02_CompareReconstructions.ipynb` | Quick pickle QA |
| `03_InspectGridOutputs.ipynb` | Unpack/inspect jobsub `out_*.tgz` campaigns |
| `05_AssessHandscanGrid.ipynb` | ROC on handscan grid campaigns |
| `06_BaselineVAE_CAE_SBND.ipynb` | VAE/CAE SBND handscan baselines |
| `07_BaselineVAE_CAE_ICARUS.ipynb` | VAE/CAE ICARUS baselines (after training) |
| `run_autoencoder_inference.py` | Batch VAE/CAE → `*_T0_ae_arrays.npz` |
| `run_ddim2ddim_inference.py` | Batch DDIM→DDIM |
| `ad_metrics.py` | AD scores (weighted MSE, latent L2, denoise loss, typicality, …) |
| `submit_ddim_grid.py` | jobsub campaign submitter (NPZ inputs) |
| `GRID.md` | Grid / pnfs how-to |
| `run_mixed_diffusion_inference.py` | rand2ddim / rand2ddpm / … |
| `run_handscan_validation_inference.py` | Handscan healthy/unhealthy set |
| `parallel_handscan_inference.py` | Memory-budgeted parallel runner |
| `CompareReconstructions-ICARUS.ipynb` | Legacy interactive compare |

From repo root:

```bash
PYTHONPATH=_stubs:train/diffusion-anomaly:inference \
  python inference/run_ddim2ddim_inference.py --help

PYTHONPATH=_stubs:train/diffusion-anomaly:inference \
  python inference/run_autoencoder_inference.py --help
```
