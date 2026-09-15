# Inference / translation

| File | Purpose |
|------|---------|
| `01_RunInference.ipynb` | EAF launcher for DDIM / mixed modes |
| `02_CompareReconstructions.ipynb` | Quick pickle QA |
| `run_ddim2ddim_inference.py` | Batch DDIM→DDIM |
| `run_mixed_diffusion_inference.py` | rand2ddim / rand2ddpm / … |
| `run_handscan_validation_inference.py` | Handscan healthy/unhealthy set |
| `parallel_handscan_inference.py` | Memory-budgeted parallel runner |
| `CompareReconstructions-ICARUS.ipynb` | Legacy interactive compare |

From repo root:

```bash
PYTHONPATH=_stubs:train/diffusion-anomaly:inference \
  python inference/run_ddim2ddim_inference.py --help
```
