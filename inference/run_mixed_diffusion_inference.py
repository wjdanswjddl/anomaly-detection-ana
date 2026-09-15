#!/usr/bin/env python3
"""
Run mixed noise → denoise pipelines on all *.npz samples under a directory
(same patching, batching, and artifact layout as ``run_ddim2ddim_inference.py``).

Default output layout
---------------------
Writes under ``<--output-dir>/mixed_<MODE>/`` so results never share a directory
with ``run_ddim2ddim_inference.py`` (which writes ``*_ddim2ddim*.pkl`` next to
``manifest_T{T}.json`` in the folder you pass). Use ``--flat-output`` only if
you intentionally want a custom layout.

Per source NPZ (same fields as the DDIM2DDIM script, with mode-specific names)
-----------------------------------------------------------------------------
- Pickle: integer patch keys → ``original``, ``<MODE>-T{T}``, ``saliency-T{T}``,
  plus ``__<MODE>_source__`` (``input_filename``, ``input_path``,
  ``input_relative_to_scan_root``, ``mode``).
- Optional ``*_<MODE>_arrays.npz`` (``original``, ``reconstructed``, ``saliency``, …).
- ``*_T{T}_meta.json`` when using the default ``mixed_<MODE>/`` directory (same
  basename pattern as ``run_ddim2ddim_inference``). With ``--flat-output``, the
  meta file is ``*_T{T}_{MODE}_meta.json`` so it cannot clobber a DDIM2DDIM
  ``*_meta.json`` in the same folder.

Modes (``--mode``)
------------------
**rand2ddim** — ``q_sample`` at ``T`` then DDIM denoise (``rand_2_ddim`` in notebooks).

**rand2ddpm** / **ddpm2ddpm** — ``q_sample`` at ``T`` then ``p_sample_loop_progressive``
(same numerics; choose name for pickle keys / filenames).

**ddim2ddpm** — DDIM forward noise then DDPM denoise (``ddim_2_ddpm``).
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

# Repository layout: train/diffusion-anomaly/guided_diffusion/
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DIFFUSION_ROOT = _REPO_ROOT / "train" / "diffusion-anomaly"
sys.path.insert(0, str(_DIFFUSION_ROOT))

from guided_diffusion import dist_util  # noqa: E402

from run_ddim2ddim_inference import (  # noqa: E402
    _json_ready,
    _parse_hyp_overrides,
    build_model_and_diffusion,
    gather_patches_from_npz,
    visualize_np,
)


def resolve_output_root(output_dir: Path, mode: str, flat: bool) -> Path:
    """
    Default ``mixed_<mode>/`` under ``output_dir`` so this script never writes
    alongside ``*_ddim2ddim.pkl`` from ``run_ddim2ddim_inference.py`` unless you
    ``--flat-output``.
    """
    out = output_dir.expanduser().resolve()
    if flat:
        return out
    return out / f"mixed_{mode}"


def ddpm_forward_noise(diffusion, x0: th.Tensor, T: int) -> th.Tensor:
    """Sample x_t from q(x_t | x_0) at timestep index T (batch-shared)."""
    dev = x0.device
    t = th.full((x0.shape[0],), int(T), device=dev, dtype=th.long)
    return diffusion.q_sample(x0, t)


def ddim_forward_noise(
    diffusion,
    model,
    x0: th.Tensor,
    T: int,
    *,
    progress: bool,
) -> th.Tensor:
    gen = diffusion.ddim_sample_loop_progressive(
        model,
        x0.shape,
        time=T,
        noise=x0,
        reverse=True,
        progress=progress,
    )
    return list(gen)[-1]["sample"]


def ddim_denoise(
    diffusion,
    model,
    shape: tuple,
    noise: th.Tensor,
    T: int,
    *,
    progress: bool,
) -> th.Tensor:
    gen = diffusion.ddim_sample_loop_progressive(
        model,
        shape,
        time=T,
        noise=noise,
        progress=progress,
    )
    return list(gen)[-1]["sample"]


def ddpm_denoise(
    diffusion,
    model,
    shape: tuple,
    noise: th.Tensor,
    T: int,
    *,
    progress: bool,
) -> th.Tensor:
    gen = diffusion.p_sample_loop_progressive(
        model,
        shape,
        time=T,
        noise=noise,
        progress=progress,
    )
    return list(gen)[-1]["sample"]


def reconstruct(
    mode: str,
    diffusion,
    model,
    imgs: th.Tensor,
    T: int,
    *,
    progress_ddim: bool,
    progress_ddpm: bool,
) -> th.Tensor:
    if mode == "rand2ddim":
        noisy = ddpm_forward_noise(diffusion, imgs, T)
        return ddim_denoise(diffusion, model, imgs.shape, noisy, T, progress=progress_ddim)
    if mode in ("rand2ddpm", "ddpm2ddpm"):
        noisy = ddpm_forward_noise(diffusion, imgs, T)
        return ddpm_denoise(diffusion, model, imgs.shape, noisy, T, progress=progress_ddpm)
    if mode == "ddim2ddpm":
        noisy = ddim_forward_noise(diffusion, model, imgs, T, progress=progress_ddim)
        return ddpm_denoise(diffusion, model, imgs.shape, noisy, T, progress=progress_ddpm)
    raise ValueError(f"Unknown mode {mode!r}")


def run_file(
    npz_path: Path,
    out_dir: Path,
    diffusion,
    model,
    *,
    mode: str,
    scan_root: Path | None,
    T: int,
    batch_size: int,
    reco_key: str,
    patch_h: int,
    patch_w: int,
    scale_div: float | None,
    write_pickle: bool,
    write_npz: bool,
    model_path_display: str,
    progress_ddim: bool,
    progress_ddpm: bool,
    meta_filename_include_mode: bool,
) -> dict:
    """Process one NPZ → outputs with global patch indexing (same contract as DDIM2DDIM script)."""

    pts, layouts, frame_slices, frame_stems = gather_patches_from_npz(
        npz_path,
        reco_key=reco_key,
        patch_h=patch_h,
        patch_w=patch_w,
        scale_div=scale_div,
    )

    if pts.shape[0] == 0:
        raise ValueError(f"No patches extracted from {npz_path}")

    n_frames = len(frame_slices)
    n_patches = int(pts.shape[0])
    print(
        f"{npz_path.name}: {n_frames} frame(s) in NPZ reco → "
        f"{n_patches} patch image(s) ({patch_h}×{patch_w}) through the model "
        f"[{mode}]"
    )

    pts = visualize_np(np.expand_dims(pts, axis=1)).astype(np.float32)

    reco_key_out = f"{mode}-T{T}"
    saliency_key = f"saliency-T{T}"
    trace_key = f"__{mode}_source__"

    save_data: dict = {}
    all_orig_np: list[np.ndarray] = []
    all_reco_np: list[np.ndarray] = []
    all_saliency_np: list[np.ndarray] = []

    dev = dist_util.dev()
    n_total = pts.shape[0]
    n_batches = (n_total + batch_size - 1) // batch_size

    for bstart in tqdm(
        range(0, n_total, batch_size),
        desc=f"{npz_path.name} [bs={batch_size}]",
        total=n_batches,
        unit="batch",
        leave=False,
    ):
        bend = min(bstart + batch_size, n_total)
        batch = pts[bstart:bend]
        imgs = th.tensor(batch, device=dev)

        reco = reconstruct(
            mode,
            diffusion,
            model,
            imgs,
            T,
            progress_ddim=progress_ddim,
            progress_ddpm=progress_ddpm,
        )

        for i in range(len(imgs)):
            gidx = bstart + i
            origin = imgs[i].detach().cpu().numpy()
            r = reco[i].detach().cpu().numpy()
            save_data[gidx] = {
                "original": origin,
                reco_key_out: r,
                saliency_key: r - origin,
            }
            all_orig_np.append(origin)
            all_reco_np.append(r)
            all_saliency_np.append(r - origin)

    stacked_orig = np.stack(all_orig_np, axis=0) if all_orig_np else np.zeros((0, 1, patch_h, patch_w))
    stacked_reco = np.stack(all_reco_np, axis=0) if all_reco_np else np.zeros_like(stacked_orig)
    stacked_sal = np.stack(all_saliency_np, axis=0) if all_saliency_np else np.zeros_like(stacked_orig)

    rel_to_scan = None
    if scan_root is not None:
        try:
            rel_to_scan = str(npz_path.resolve().relative_to(scan_root.resolve()))
        except ValueError:
            rel_to_scan = None

    trace = {
        "input_filename": npz_path.name,
        "input_path": str(npz_path.resolve()),
        "input_relative_to_scan_root": rel_to_scan,
        "mode": mode,
    }

    meta = {
        **trace,
        "T": T,
        "model_path": model_path_display,
        "source_npz": str(npz_path.resolve()),
        "reco_key": reco_key,
        "n_patches": int(n_total),
        "patch_h": patch_h,
        "patch_w": patch_w,
        "frame_slices": [{"stem": stem, "start": a, "end": b} for stem, (a, b) in zip(frame_stems, frame_slices)],
        "layouts_per_frame": layouts,
        "pickle_keys_per_patch": ["original", reco_key_out, saliency_key],
        "formats": [],
    }

    stem = npz_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    if write_pickle:
        pkl_path = out_dir / f"{stem}_T{T}_{mode}.pkl"
        meta["formats"].append("pickle")
        save_data[trace_key] = dict(trace)
        with open(pkl_path, "wb") as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    if write_npz:
        nz_path = out_dir / f"{stem}_T{T}_{mode}_arrays.npz"
        meta["formats"].append("npz_arrays")
        np.savez_compressed(
            nz_path,
            original=stacked_orig.astype(np.float32, copy=False),
            reconstructed=stacked_reco.astype(np.float32, copy=False),
            saliency=stacked_sal.astype(np.float32, copy=False),
            input_filename=np.asarray(trace["input_filename"]),
            input_path=np.asarray(trace["input_path"]),
            input_relative_to_scan_root=np.asarray(trace["input_relative_to_scan_root"] or ""),
            mode=np.asarray(mode),
        )

    if meta_filename_include_mode:
        json_path = out_dir / f"{stem}_T{T}_{mode}_meta.json"
    else:
        json_path = out_dir / f"{stem}_T{T}_meta.json"
    with open(json_path, "w") as wf:
        json.dump(_json_ready(meta), wf, indent=2)

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory of *.npz files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Parent directory for this run. By default artifacts go in "
            "``OUTPUT_DIR/mixed_<MODE>/`` (distinct from ``run_ddim2ddim_inference``). "
            "Pass ``--flat-output`` to use ``OUTPUT_DIR`` itself."
        ),
    )
    parser.add_argument("--model-path", type=Path, required=True, help="Checkpoint .pt (EMA update)")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=("rand2ddim", "rand2ddpm", "ddpm2ddpm", "ddim2ddpm"),
        help="Noise / denoise recipe (ddpm2ddpm matches rand2ddpm numerically; tag differs in keys/filenames)",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help=(
            "Write pickle/npz/meta directly under --output-dir instead of mixed_<MODE>/. "
            "Only use if you keep DDIM2DDIM outputs elsewhere."
        ),
    )
    parser.add_argument(
        "--T",
        type=int,
        required=True,
        dest="timestep_T",
        help="Noise level timestep (matches notebook ``T`` in ddim / p_sample loops)",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--reco-key", type=str, default="reco")
    parser.add_argument("--patch-size", type=int, default=512, help="Square patch edge length")
    parser.add_argument(
        "--scale-divisor",
        type=float,
        default=None,
        help="If set (e.g. 200.), divide plane by this constant before patching",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories for *.npz (output mirrors relative paths)",
    )
    parser.add_argument(
        "--ddim-progress",
        action="store_true",
        help=(
            "Show tqdm progress *inside each DDIM chain* (~T steps per encode/decode "
            "per batch). Slow; leave off unless debugging."
        ),
    )
    parser.add_argument(
        "--ddpm-progress",
        action="store_true",
        help=(
            "Show tqdm progress *inside each DDPM reverse chain* (~T steps per batch). "
            "Slow; leave off unless debugging."
        ),
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="pickle,npz",
        help="Comma-separated: pickle, npz (``*_meta.json`` is always written)",
    )
    parser.add_argument(
        "--hyp-override",
        action="append",
        default=[],
        metavar="NAME=NUMBER",
        help="Optional overrides, e.g. diffusion_steps=1000 noise_schedule=cosine image_size=512",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N *.npz files (useful for quick tests)",
    )

    parsed = parser.parse_args()
    input_dir = parsed.input_dir.expanduser().resolve()
    run_root = resolve_output_root(parsed.output_dir, parsed.mode, parsed.flat_output)
    model_path = parsed.model_path.expanduser().resolve()

    if not input_dir.is_dir():
        sys.exit(f"Not a directory: {input_dir}")
    if not model_path.is_file():
        sys.exit(f"No checkpoint file: {model_path}")

    glober = input_dir.rglob if parsed.recursive else input_dir.glob
    npz_paths = sorted(glober("*.npz"))
    if not npz_paths:
        sys.exit(f"No *.npz under {input_dir}")
    if parsed.max_files is not None:
        npz_paths = npz_paths[: parsed.max_files]

    fmts = [x.strip().lower() for x in parsed.formats.split(",")]
    write_pickle = "pickle" in fmts
    write_npz = "npz" in fmts
    patch_h = patch_w = int(parsed.patch_size)
    scale_div: float | None = parsed.scale_divisor

    run_root.mkdir(parents=True, exist_ok=True)
    if not parsed.flat_output:
        print(f"Output root (mixed, distinct from ddim2ddim): {run_root}")

    th.set_grad_enabled(False)
    hypextra = _parse_hyp_overrides(parsed.hyp_override)
    model, diffusion = build_model_and_diffusion(hypextra)
    sd = dist_util.load_state_dict(str(model_path), map_location="cpu")
    model.load_state_dict(sd)
    model.to(dist_util.dev())
    model.eval()

    manifest_items: list[dict] = []

    for nz in npz_paths:
        if parsed.recursive:
            rel_par = nz.parent.relative_to(input_dir)
            leaf_out = run_root / rel_par
        else:
            leaf_out = run_root

        meta_run = run_file(
            nz,
            leaf_out,
            diffusion,
            model,
            mode=parsed.mode,
            scan_root=input_dir,
            T=parsed.timestep_T,
            batch_size=parsed.batch_size,
            reco_key=parsed.reco_key,
            patch_h=patch_h,
            patch_w=patch_w,
            scale_div=scale_div,
            write_pickle=write_pickle,
            write_npz=write_npz,
            model_path_display=str(model_path),
            progress_ddim=parsed.ddim_progress,
            progress_ddpm=parsed.ddpm_progress,
            meta_filename_include_mode=parsed.flat_output,
        )
        manifest_items.append(meta_run)

    sweep_json = run_root / f"manifest_T{parsed.timestep_T}.json"
    with open(sweep_json, "w") as wf:
        json.dump(
            _json_ready(
                {
                    "T": parsed.timestep_T,
                    "mode": parsed.mode,
                    "run_root": str(run_root),
                    "flat_output": bool(parsed.flat_output),
                    "model_path": str(model_path),
                    "runs": manifest_items,
                }
            ),
            wf,
            indent=2,
        )


if __name__ == "__main__":
    main()
