#!/usr/bin/env python3
"""
Run VAE/CAE reconstruction anomaly detection on NPZ (or single-file) inputs.

Writes products compatible with the diffusion handscan ROC pipeline:

- ``{stem}_T0_ae_arrays.npz`` with ``original``, ``reconstructed``, ``saliency``
  (``saliency = original - reconstructed``, same sign convention as the demos)
- optional pickle / meta / QA plots

``T0`` is a placeholder so scoring notebooks that expect ``_T{T}_`` in the
filename still work; autoencoders have no diffusion timestep.

Example (CPU-safe smoke)::

    PYTHONPATH=_stubs:train/diffusion-anomaly:inference \\
      python inference/run_autoencoder_inference.py \\
        --ae-type vae --model-path /path/to/ema.pt \\
        --input-dir .../handscan/grid/npz_inference/healthy \\
        --output-dir .../inference/baselines/vae_sbnd --device cpu --max-files 1
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch as th
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DIFFUSION_ROOT = _REPO_ROOT / "train" / "diffusion-anomaly"
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_DIFFUSION_ROOT))

from guided_diffusion.script_util import create_autoencoder  # noqa: E402

from configs.paths import ae_model_config  # noqa: E402

# Reuse NPZ tiling helpers from the DDIM runner.
from run_ddim2ddim_inference import (  # noqa: E402
    load_reco_array,
    patches_from_plane,
    visualize_np,
)


def load_ae_model(ae_type: str, model_path: Path, device: th.device):
    cfg = ae_model_config(ae_type)
    model = create_autoencoder(**cfg)
    state = th.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, cfg


def reconstruct_batch(
    model,
    patches: np.ndarray,
    *,
    device: th.device,
    batch_size: int,
) -> np.ndarray:
    """patches: (N, H, W) or (N, 1, H, W) → reconstructed (N, 1, H, W)."""
    if patches.ndim == 3:
        patches = patches[:, None, :, :]
    out = []
    with th.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = th.from_numpy(patches[i : i + batch_size].astype(np.float32)).to(device)
            reco = model.reconstruct(batch)
            out.append(reco.cpu().numpy())
    return np.concatenate(out, axis=0)


def process_npz(
    npz_path: Path,
    model,
    *,
    device: th.device,
    out_dir: Path,
    patch: int,
    batch_size: int,
    reco_key: str,
    write_pickle: bool,
) -> Path:
    frames = load_reco_array(npz_path, reco_key=reco_key)
    # frames: (F, H, W) or (F, 1, H, W)
    if frames.ndim == 4:
        frames = frames[:, 0]
    all_orig: list[np.ndarray] = []
    all_reco: list[np.ndarray] = []
    all_sal: list[np.ndarray] = []
    frame_slices: list[dict] = []
    results: dict = {
        "__ae_source__": {
            "input_filename": npz_path.name,
            "input_path": str(npz_path),
            "ae_type": getattr(model, "ae_type", None),
        }
    }
    patch_i = 0
    for fi, plane in enumerate(frames):
        plane = visualize_np(plane.astype(np.float32))
        patches, layout = patches_from_plane(plane, patch, patch)
        reco = reconstruct_batch(model, patches, device=device, batch_size=batch_size)
        orig = patches[:, None, :, :].astype(np.float32)
        sal = orig - reco
        n = orig.shape[0]
        for j in range(n):
            results[patch_i + j] = {
                "original": orig[j],
                "ae": reco[j],
                "saliency": sal[j],
            }
        frame_slices.append(
            {
                "frame": fi,
                "patch_start": patch_i,
                "patch_end": patch_i + n,
                **{k: list(v) if isinstance(v, tuple) else v for k, v in layout.items()},
            }
        )
        patch_i += n
        all_orig.append(orig)
        all_reco.append(reco)
        all_sal.append(sal)

    stem = npz_path.stem
    stacked_o = np.concatenate(all_orig, axis=0)
    stacked_r = np.concatenate(all_reco, axis=0)
    stacked_s = np.concatenate(all_sal, axis=0)
    arrays_path = out_dir / f"{stem}_T0_ae_arrays.npz"
    np.savez_compressed(
        arrays_path,
        original=stacked_o,
        reconstructed=stacked_r,
        saliency=stacked_s,
    )
    meta = {
        "stem": stem,
        "source": str(npz_path),
        "n_patches": int(stacked_o.shape[0]),
        "frame_slices": frame_slices,
        "score_hint": "file_max_rms of saliency",
        "mode": "ae",
        "T": 0,
    }
    (out_dir / f"{stem}_T0_ae_meta.json").write_text(json.dumps(meta, indent=2))
    if write_pickle:
        with open(out_dir / f"{stem}_T0_ae.pkl", "wb") as f:
            pickle.dump(results, f)
    return arrays_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ae-type", choices=("vae", "cae"), required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--input-dir", type=Path, default=None, help="Directory of *.npz inputs")
    p.add_argument("--input", type=Path, default=None, help="Single .npz / .h5 file")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--patch", type=int, default=512)
    p.add_argument("--reco-key", default="reco")
    p.add_argument("--max-files", type=int, default=0, help="0 = all")
    p.add_argument("--write-pickle", action="store_true")
    args = p.parse_args()

    if (args.input_dir is None) == (args.input is None):
        p.error("Provide exactly one of --input-dir or --input")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = th.device(args.device)
    print(f"device={device}  ae_type={args.ae_type}  model={args.model_path}")

    model, cfg = load_ae_model(args.ae_type, args.model_path, device)
    (out_dir / "model_config.json").write_text(json.dumps(cfg, indent=2))

    if args.input is not None:
        paths = [args.input]
    else:
        paths = sorted(args.input_dir.rglob("*.npz"))
        if args.max_files > 0:
            paths = paths[: args.max_files]

    wrote = []
    for path in tqdm(paths, desc="ae-inference"):
        if path.suffix.lower() == ".npz":
            wrote.append(
                process_npz(
                    path,
                    model,
                    device=device,
                    out_dir=out_dir,
                    patch=args.patch,
                    batch_size=args.batch_size,
                    reco_key=args.reco_key,
                    write_pickle=args.write_pickle,
                )
            )
        else:
            raise SystemExit(f"Unsupported input (use .npz via this runner): {path}")

    print(f"wrote {len(wrote)} array files under {out_dir}")


if __name__ == "__main__":
    main()
