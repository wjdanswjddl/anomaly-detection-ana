#!/usr/bin/env python3
"""
Run DDIM-forward → DDIM-inverse reconstruction on all *.npz samples under a directory
(same diffusion recipe as CompareReconstructions-ICARUS.ipynb).

Writes, per source file and per T:

- Pickle (reco_jobs / ROC layout): integer keys → per-patch dict with ``original``,
  ``diffused-T{T}`` (DDIM-encoded latent at T), ``ddim2ddim-T{T}``, and
  ``saliency-T{T}`` (reconstruction − original).
  Arrays NPZ also stores ``diffused`` alongside ``original`` / ``reconstructed`` /
  ``saliency``.

Pass multiple ``--T`` values to sweep (patches extracted once; each T is written
separately as ``*_T{T}_ddim2ddim.pkl`` / arrays / meta).

- Same pickle carries ``__ddim2ddim_source__`` → ``input_filename``, ``input_path``, and
  ``input_relative_to_scan_root`` (when the file sits under ``--input-dir``) so you can trace
  results back without relying on output filenames.

- Optional stacked arrays ``*_ddim2ddim_arrays.npz``, and ``*_meta.json`` with matching fields.
- Optional AD metrics ``*_T{T}_ddim2ddim_ad_metrics.npz`` (weighted MSE, latent L2,
  denoise loss, posterior typicality, …) — see ``inference/ad_metrics.py``.

ICARUS-style NPZ arrays use ``reco`` shaped like (frames, 1, H, W), (frames, H, W), or (H, W).
Planes are cropped to multiples of patch size and tiled into model-sized patches.

Reassembly for visualization: ``stitch_patches`` and ``*_meta.json`` ``frame_slices``.

ROC-style scores (patch index ``iidx``); skip the string traceability key::

    with open(run_dir / f"stem_T{T}_ddim2ddim.pkl", "rb") as f:
        results = pickle.load(f)
    original_name = results["__ddim2ddim_source__"]["input_filename"]
    dk = f"ddim2ddim-T{T}"
    for iidx in sorted(k for k in results if isinstance(k, int)):
        diff = results[iidx][dk][0] - results[iidx]["original"][0]
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
from guided_diffusion.script_util import (  # noqa: E402
    create_model_and_diffusion,
    diffusion_defaults,
    model_and_diffusion_defaults,
)


_SKIP_FILENAME_SUBSTRINGS = ("tpc2-plane1", "tpc1-plane1")


def _skip_npz(npz_path: Path) -> bool:
    name = npz_path.name
    return any(s in name for s in _SKIP_FILENAME_SUBSTRINGS)


def visualize_np(img: np.ndarray) -> np.ndarray:
    """Match notebook: clip to diffusion value range."""
    return np.clip(img, -1.0, 1.0)


def stitch_patches(
    patches: np.ndarray, grid_shape: tuple[int, int], patch_size: tuple[int, int] | None = None
) -> np.ndarray:
    """Rebuild a plane from row-major patches (same as CompareReconstructions-ICARUS)."""
    nh, nw = grid_shape
    if patch_size is None:
        ph, pw = int(patches.shape[-2]), int(patches.shape[-1])
    else:
        ph, pw = patch_size

    if patches.ndim == 4:
        n, c, _, _ = patches.shape
        return (
            patches.reshape(nh, nw, c, ph, pw).transpose(0, 2, 1, 3, 4).reshape(c, nh * ph, nw * pw)
        )
    if patches.ndim == 3:
        return patches.reshape(nh, nw, ph, pw).swapaxes(1, 2).reshape(nh * ph, nw * pw)
    raise ValueError(f"Expected 3D or 4D array of patches, got {patches.ndim}D")


def patches_from_plane(
    planearr: np.ndarray,
    patch_h: int,
    patch_w: int,
) -> tuple[np.ndarray, dict]:
    """Return (N_patches, patch_h, patch_w) patches and patch_layout metadata dict."""
    h, w = planearr.shape
    hc = (h // patch_h) * patch_h
    wc = (w // patch_w) * patch_w
    cropped = planearr[:hc, :wc]
    nh, nw = hc // patch_h, wc // patch_w
    if nh == 0 or nw == 0:
        raise ValueError(
            f"Cropped plane {hc}x{wc} is smaller than patch {patch_h}x{patch_w} "
            f"(full shape {h}x{w})"
        )
    ll = (
        cropped.reshape(nh, patch_h, nw, patch_w).swapaxes(1, 2).reshape(-1, patch_h, patch_w)
    )
    layout = {
        "patch_size": (patch_h, patch_w),
        "grid_shape": (nh, nw),
        "cropped_shape": (hc, wc),
        "full_plane_shape": (h, w),
    }
    return ll, layout


def load_reco_array(npz_path: Path, reco_key: str = "reco") -> np.ndarray:
    z = np.load(npz_path, allow_pickle=True)
    if reco_key not in z.files:
        raise KeyError(f"{npz_path}: no '{reco_key}' — available {z.files}")
    arr = z[reco_key].astype(np.float32, copy=False)
    z.close()

    # Normalize to list of (H,W) planes
    if arr.ndim == 2:
        return arr[np.newaxis, ...]

    if arr.ndim == 3:
        # (frames, H, W)
        return arr

    if arr.ndim == 4:
        # (frames, C, H, W) → require C == 1
        if arr.shape[1] != 1:
            raise ValueError(f"{npz_path}: reco has C={arr.shape[1]}, expected 1")
        return arr[:, 0, :, :]

    raise ValueError(f"{npz_path}: unexpected reco shape {arr.shape}")


def gather_patches_from_npz(
    npz_path: Path,
    reco_key: str,
    patch_h: int,
    patch_w: int,
    scale_div: float | None,
) -> tuple[np.ndarray, list[dict], list[tuple[int, int]], list[str]]:
    """
    Flatten every frame into 512² patches.

    Returns
    -------
    patches : (total_patches, patch_h, patch_w)
    layouts : layout dict per contiguous frame-run (here one per frame)
    frame_slices : parallel list (start, exclusive_end) indexing rows of ``patches``
    stems : descriptive id per slice (frame index encoded as string)
    """
    reco_stack = load_reco_array(npz_path, reco_key)
    patches_list: list[np.ndarray] = []
    layouts_list: list[dict] = []
    frame_slices: list[tuple[int, int]] = []
    stems: list[str] = []

    cursor = 0
    for i in range(len(reco_stack)):
        plane = reco_stack[i]
        if scale_div is not None and scale_div != 0:
            plane = plane / scale_div
        ppt, lay = patches_from_plane(plane, patch_h, patch_w)
        n_here = ppt.shape[0]
        patches_list.append(ppt)
        layouts_list.append(lay | {"npz_frame": int(i)})
        frame_slices.append((cursor, cursor + n_here))
        stems.append(f"frame{i:04d}_{lay['grid_shape'][0]}x{lay['grid_shape'][1]}patches")
        cursor += n_here

    patches = np.concatenate(patches_list, axis=0) if patches_list else np.zeros((0, patch_h, patch_w))
    return patches, layouts_list, frame_slices, stems


def build_model_and_diffusion(overrides: dict | None = None):
    """Match CompareReconstructions-ICARUS / reco_jobs hyperparameters.

    ``overrides`` may include training-config keys, e.g.
    ``noise_schedule``, ``predict_xstart``, ``anisotropic_noise``,
    ``noise_mode``, ``empty_noise_fraction``, ``smoothing_sigma``.
    """
    overrides = dict(overrides or {})
    args = model_and_diffusion_defaults()
    diffusion_args = diffusion_defaults()

    args["image_size"] = int(overrides.get("image_size", 512))
    args["num_channels"] = int(overrides.get("num_channels", 32))
    args["class_cond"] = False
    args["num_res_blocks"] = int(overrides.get("num_res_blocks", 2))
    args["num_heads"] = int(overrides.get("num_heads", 8))
    args["learn_sigma"] = True
    args["use_scale_shift_norm"] = False
    args["attention_resolutions"] = str(overrides.get("attention_resolutions", "16,32"))
    args["channel_mult"] = str(overrides.get("channel_mult", "1,2,4,8,8,8"))

    diffusion_args["diffusion_steps"] = int(overrides.get("diffusion_steps", 1000))
    diffusion_args["noise_schedule"] = str(overrides.get("noise_schedule", "linear"))
    diffusion_args["rescale_learned_sigmas"] = False
    diffusion_args["rescale_timesteps"] = False
    diffusion_args["predict_xstart"] = bool(overrides.get("predict_xstart", False))
    if "anisotropic_noise" in diffusion_args or "anisotropic_noise" in overrides:
        diffusion_args["anisotropic_noise"] = bool(overrides.get("anisotropic_noise", False))
    for k in (
        "noise_mode",
        "empty_noise_fraction",
        "smoothing_sigma",
        "occupancy_threshold",
        "local_std_patch_size",
    ):
        if k in overrides:
            diffusion_args[k] = overrides[k]
    diffusion_args.pop("diffusion_steps", None)
    diffusion_args.pop("timestep_respacing", None)
    diffusion_args["learn_sigma"] = True

    args = args | diffusion_args

    model, diffusion = create_model_and_diffusion(**args)
    return model, diffusion


def overrides_for_train_config(name: str) -> dict:
    """Map ``configs.train_configs`` name → ``build_model_and_diffusion`` overrides."""
    from configs.train_configs import TRAIN_CONFIG_BY_NAME

    c = TRAIN_CONFIG_BY_NAME[name]
    return {
        "noise_schedule": c.noise_schedule,
        "predict_xstart": c.predict_xstart,
        "anisotropic_noise": c.anisotropic_noise,
    }


def ddim2ddim_reconstruct(
    diffusion,
    model,
    imgs_btch: th.Tensor,
    T: int,
    *,
    progress: bool,
) -> tuple[th.Tensor, th.Tensor]:
    """
    imgs_btch shape (N, 1, H, W), same semantics as notebook:
    DDIM noisy (reverse loop) → DDIM denoise back to estimate of x_0.
    """
    ddim_noise_gen = diffusion.ddim_sample_loop_progressive(
        model,
        imgs_btch.shape,
        time=T,
        noise=imgs_btch,
        reverse=True,
        progress=progress,
    )
    ddim_noised = list(ddim_noise_gen)[-1]["sample"]

    ddim_den = diffusion.ddim_sample_loop_progressive(
        model,
        imgs_btch.shape,
        time=T,
        noise=ddim_noised,
        progress=progress,
    )
    reco = list(ddim_den)[-1]["sample"]
    return ddim_noised, reco


def _json_ready(obj):
    """Convert tuples/numpy ints for json.dump."""
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(x) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def run_file(
    npz_path: Path,
    out_dir: Path,
    diffusion,
    model,
    *,
    scan_root: Path | None,
    T: int | list[int],
    batch_size: int,
    reco_key: str,
    patch_h: int,
    patch_w: int,
    scale_div: float | None,
    write_pickle: bool,
    write_npz: bool,
    model_path_display: str,
    progress_ddim: bool,
    ad_metrics: bool = True,
    ad_posterior_k: int = 0,
    ad_t_loss: int | None = None,
    ad_t_embed: int = 0,
) -> list[dict]:
    """Process one NPZ → per-T outputs with global patch indexing (ROC notebooks).

    ``T`` may be a single int or a list (sweep). Patches are extracted once; each
    T writes its own ``*_T{T}_ddim2ddim.pkl`` / arrays / meta (same layout as before).
    """
    from ad_metrics import compute_ad_metrics_batch, save_ad_metrics_npz

    if isinstance(T, int):
        timesteps = [int(T)]
    else:
        timesteps = [int(t) for t in T]
    if not timesteps:
        raise ValueError("Need at least one T")
    # Preserve order, drop duplicates
    seen: set[int] = set()
    timesteps = [t for t in timesteps if not (t in seen or seen.add(t))]
    if any(t <= 0 for t in timesteps):
        raise ValueError(f"T values must be positive, got {timesteps}")

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
        f"{n_patches} patch image(s) ({patch_h}×{patch_w}) through the model; "
        f"T sweep={timesteps}"
    )

    pts = visualize_np(np.expand_dims(pts, axis=1)).astype(np.float32)

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
    }

    stem = npz_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = dist_util.dev()
    n_total = pts.shape[0]
    n_batches = (n_total + batch_size - 1) // batch_size
    metas: list[dict] = []

    for T_i in timesteps:
        save_data: dict = {}
        all_orig_np: list[np.ndarray] = []
        all_diffused_np: list[np.ndarray] = []
        all_reco_np: list[np.ndarray] = []
        all_saliency_np: list[np.ndarray] = []

        ddim_key = f"ddim2ddim-T{T_i}"
        saliency_key = f"saliency-T{T_i}"

        for bstart in tqdm(
            range(0, n_total, batch_size),
            desc=f"{npz_path.name} T={T_i} [bs={batch_size}]",
            total=n_batches,
            unit="batch",
            leave=False,
        ):
            bend = min(bstart + batch_size, n_total)
            batch = pts[bstart:bend]
            imgs = th.tensor(batch, device=dev)

            ddim_noised, reco = ddim2ddim_reconstruct(
                diffusion, model, imgs, T_i, progress=progress_ddim
            )

            for i in range(len(imgs)):
                gidx = bstart + i
                origin = imgs[i].detach().cpu().numpy()
                noisy = ddim_noised[i].detach().cpu().numpy()
                r = reco[i].detach().cpu().numpy()
                save_data[gidx] = {
                    "original": origin,
                    f"diffused-T{T_i}": noisy,
                    ddim_key: r,
                    saliency_key: r - origin,
                }
                all_orig_np.append(origin)
                all_diffused_np.append(noisy)
                all_reco_np.append(r)
                all_saliency_np.append(r - origin)

        stacked_orig = (
            np.stack(all_orig_np, axis=0)
            if all_orig_np
            else np.zeros((0, 1, patch_h, patch_w))
        )
        stacked_diffused = (
            np.stack(all_diffused_np, axis=0)
            if all_diffused_np
            else np.zeros_like(stacked_orig)
        )
        stacked_reco = (
            np.stack(all_reco_np, axis=0) if all_reco_np else np.zeros_like(stacked_orig)
        )
        stacked_sal = (
            np.stack(all_saliency_np, axis=0)
            if all_saliency_np
            else np.zeros_like(stacked_orig)
        )

        meta = {
            **trace,
            "T": T_i,
            "T_sweep": list(timesteps),
            "model_path": model_path_display,
            "source_npz": str(npz_path.resolve()),
            "reco_key": reco_key,
            "n_patches": int(n_total),
            "patch_h": patch_h,
            "patch_w": patch_w,
            "frame_slices": [
                {"stem": stem_f, "start": a, "end": b}
                for stem_f, (a, b) in zip(frame_stems, frame_slices)
            ],
            "layouts_per_frame": layouts,
            "pickle_keys_per_patch": [
                "original",
                f"diffused-T{T_i}",
                ddim_key,
                saliency_key,
            ],
            "formats": [],
            "ad_metrics": bool(ad_metrics),
            "ad_posterior_k": int(ad_posterior_k),
        }

        if write_pickle:
            pkl_path = out_dir / f"{stem}_T{T_i}_ddim2ddim.pkl"
            meta["formats"].append("pickle")
            save_data["__ddim2ddim_source__"] = dict(trace)
            with open(pkl_path, "wb") as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        if write_npz:
            nz_path = out_dir / f"{stem}_T{T_i}_ddim2ddim_arrays.npz"
            meta["formats"].append("npz_arrays")
            np.savez_compressed(
                nz_path,
                original=stacked_orig.astype(np.float32, copy=False),
                diffused=stacked_diffused.astype(np.float32, copy=False),
                reconstructed=stacked_reco.astype(np.float32, copy=False),
                saliency=stacked_sal.astype(np.float32, copy=False),
                input_filename=np.asarray(trace["input_filename"]),
                input_path=np.asarray(trace["input_path"]),
                input_relative_to_scan_root=np.asarray(
                    trace["input_relative_to_scan_root"] or ""
                ),
            )

        if ad_metrics and n_total > 0:
            metrics = compute_ad_metrics_batch(
                stacked_orig,
                stacked_reco,
                model=model,
                diffusion=diffusion,
                T=T_i,
                t_loss=ad_t_loss if ad_t_loss is not None else T_i,
                t_embed=ad_t_embed,
                posterior_k=int(ad_posterior_k),
                progress_posterior=False,
            )
            mpath = out_dir / f"{stem}_T{T_i}_ddim2ddim_ad_metrics.npz"
            save_ad_metrics_npz(mpath, metrics, trace=trace)
            meta["formats"].append("ad_metrics")
            meta["ad_metrics_path"] = mpath.name
            # compact summary for manifest
            summary = {}
            for name in metrics.get("metric_names", []):
                arr = metrics.get(name)
                if arr is None:
                    continue
                a = np.asarray(arr, dtype=np.float64)
                a = a[np.isfinite(a)]
                if a.size:
                    summary[name] = {
                        "mean": float(a.mean()),
                        "max": float(a.max()),
                        "min": float(a.min()),
                    }
            meta["ad_metrics_summary"] = summary

        json_path = out_dir / f"{stem}_T{T_i}_meta.json"
        with open(json_path, "w") as wf:
            json.dump(_json_ready(meta), wf, indent=2)

        metas.append(meta)

    return metas


STRING_HYP_KEYS = frozenset(
    {
        "noise_schedule",
        "attention_resolutions",
        "channel_mult",
        "noise_mode",
    }
)
BOOL_HYP_KEYS = frozenset(
    {
        "anisotropic_noise",
        "predict_xstart",
        "learn_sigma",
        "class_cond",
        "rescale_learned_sigmas",
        "rescale_timesteps",
        "use_scale_shift_norm",
    }
)


def _parse_hyp_overrides(items: list[str]) -> dict:
    """``key=value`` pairs for model/diffusion options."""
    out: dict = {}
    int_keys = {
        "image_size",
        "num_channels",
        "diffusion_steps",
        "num_res_blocks",
        "num_heads",
    }
    for kv in items or []:
        if "=" not in kv:
            raise ValueError(f"Expected key=value override, got {kv!r}")
        k, v_raw = kv.split("=", 1)
        key = k.strip()
        vs = v_raw.strip()
        if key in STRING_HYP_KEYS:
            out[key] = vs
            continue
        if key in BOOL_HYP_KEYS:
            out[key] = vs.lower() in ("1", "true", "yes", "y", "on")
            continue
        if "." in vs or ("e" in vs.lower() and not vs.lower().startswith("0x")):
            val = float(vs)
            out[key] = int(val) if key in int_keys else val
        else:
            try:
                out[key] = int(vs)
            except ValueError:
                out[key] = vs
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory of *.npz files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where outputs are written")
    parser.add_argument("--model-path", type=Path, required=True, help="Checkpoint .pt (EMA update)")
    parser.add_argument(
        "--T",
        type=int,
        nargs="+",
        required=True,
        dest="timestep_Ts",
        metavar="T",
        help=(
            "Noise-level timestep(s) for DDIM encode/decode. Pass one or more for a "
            "sweep, e.g. ``--T 50 100 200 400``. Each T writes separate "
            "``*_T{T}_ddim2ddim.pkl`` / arrays / meta (patches extracted once)."
        ),
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
            "Show tqdm progress *inside each DDIM chain* (~T steps per encode + ~T per decode "
            "per batch). Slow and looks like '+1 per timestep'; leave off unless debugging. "
            "Batching still applies: each outer 'batch' step runs DDIM once on all patches in batch."
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
        metavar="NAME=VALUE",
        help="Optional overrides, e.g. diffusion_steps=1000 noise_schedule=cosine anisotropic_noise=true",
    )
    parser.add_argument(
        "--train-config",
        default=None,
        choices=["linear", "cosine", "ramp", "anisotropic", "pred_xstart"],
        help="Apply schedule/architecture overrides from configs.train_configs (e.g. anisotropic).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N *.npz files (useful for quick tests)",
    )
    parser.add_argument(
        "--ad-metrics",
        dest="ad_metrics",
        action="store_true",
        default=True,
        help="Write ``*_ad_metrics.npz`` (default: on).",
    )
    parser.add_argument(
        "--no-ad-metrics",
        dest="ad_metrics",
        action="store_false",
        help="Skip AD metrics NPZ.",
    )
    parser.add_argument(
        "--ad-posterior-k",
        type=int,
        default=0,
        metavar="K",
        help=(
            "If K>0, also compute posterior typicality with K stochastic "
            "q-sample→DDIM counterfactuals (adds ~K full denoise passes). Default 0."
        ),
    )
    parser.add_argument(
        "--ad-t-loss",
        type=int,
        default=None,
        metavar="T",
        help="Timestep for denoise_loss score (default: same as reconstruction T).",
    )
    parser.add_argument(
        "--ad-t-embed",
        type=int,
        default=0,
        metavar="T",
        help="UNet timestep used for bottleneck embeddings (default: 0).",
    )

    parsed = parser.parse_args()
    input_dir = parsed.input_dir.expanduser().resolve()
    output_dir = parsed.output_dir.expanduser().resolve()
    model_path = parsed.model_path.expanduser().resolve()

    if not input_dir.is_dir():
        sys.exit(f"Not a directory: {input_dir}")
    if not model_path.is_file():
        sys.exit(f"No checkpoint file: {model_path}")

    glober = input_dir.rglob if parsed.recursive else input_dir.glob
    npz_paths = sorted(glober("*.npz"))
    npz_paths = [p for p in npz_paths if not _skip_npz(p)]
    if not npz_paths:
        sys.exit(f"No *.npz under {input_dir}")
    if parsed.max_files is not None:
        npz_paths = npz_paths[: parsed.max_files]

    hypextra: dict = {}
    if parsed.train_config:
        # Ensure repo root import works when launched from grid PYTHONPATH.
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        hypextra.update(overrides_for_train_config(parsed.train_config))
        if parsed.train_config == "anisotropic":
            hypextra.update(
                {
                    "noise_mode": "signal_proportional",
                    "empty_noise_fraction": 0.05,
                    "smoothing_sigma": 2.0,
                }
            )
    hypextra.update(_parse_hyp_overrides(parsed.hyp_override))

    fmts = [x.strip().lower() for x in parsed.formats.split(",")]
    write_pickle = "pickle" in fmts
    write_npz = "npz" in fmts

    patch_h = patch_w = int(parsed.patch_size)

    scale_div: float | None = parsed.scale_divisor

    th.set_grad_enabled(False)
    model, diffusion = build_model_and_diffusion(hypextra)
    sd = dist_util.load_state_dict(str(model_path), map_location="cpu")
    model.load_state_dict(sd)
    model.to(dist_util.dev())
    model.eval()

    timesteps = list(parsed.timestep_Ts)
    print(f"T sweep: {timesteps}")

    manifest_items: list[dict] = []
    by_T: dict[int, list[dict]] = {t: [] for t in timesteps}

    for nz in npz_paths:
        if parsed.recursive:
            rel_par = nz.parent.relative_to(input_dir)
            leaf_out = output_dir / rel_par
        else:
            leaf_out = output_dir

        metas_run = run_file(
            nz,
            leaf_out,
            diffusion,
            model,
            scan_root=input_dir,
            T=timesteps,
            batch_size=parsed.batch_size,
            reco_key=parsed.reco_key,
            patch_h=patch_h,
            patch_w=patch_w,
            scale_div=scale_div,
            write_pickle=write_pickle,
            write_npz=write_npz,
            model_path_display=str(model_path),
            progress_ddim=parsed.ddim_progress,
            ad_metrics=bool(parsed.ad_metrics),
            ad_posterior_k=int(parsed.ad_posterior_k),
            ad_t_loss=parsed.ad_t_loss,
            ad_t_embed=int(parsed.ad_t_embed),
        )
        for meta_run in metas_run:
            manifest_items.append(meta_run)
            by_T[int(meta_run["T"])].append(meta_run)

    for T_i, runs in by_T.items():
        sweep_json = output_dir / f"manifest_T{T_i}.json"
        with open(sweep_json, "w") as wf:
            json.dump(
                _json_ready({"T": T_i, "T_sweep": timesteps, "model_path": str(model_path), "runs": runs}),
                wf,
                indent=2,
            )

    combined = output_dir / "manifest.json"
    with open(combined, "w") as wf:
        json.dump(
            _json_ready(
                {
                    "T_sweep": timesteps,
                    "model_path": str(model_path),
                    "n_files": len(npz_paths),
                    "runs": manifest_items,
                }
            ),
            wf,
            indent=2,
        )
    print(f"Wrote {combined} and manifest_T*.json for T={timesteps}")


if __name__ == "__main__":
    main()
