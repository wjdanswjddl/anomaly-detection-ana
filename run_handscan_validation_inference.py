#!/usr/bin/env python3
"""Run DDIM→DDIM inference on handscan validation NPZs (healthy / unhealthy).

Writes per-file pickles compatible with AnalyzeDDIM2DDIM_Outputs / AnalyzeHandscanValidation,
plus side-by-side original / reconstruction / difference PNG plots.

Example::

    PYTHONPATH=_stubs:train/diffusion-anomaly \\
      python run_handscan_validation_inference.py \\
        --T 100 \\
        --model-path /exp/sbnd/data/users/gputnam/training-SBND/iterE/results/brats2update111000.pt
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO / "_stubs"))
sys.path.insert(0, str(_REPO / "train" / "diffusion-anomaly"))

from guided_diffusion import dist_util  # noqa: E402
from run_ddim2ddim_inference import (  # noqa: E402
    build_model_and_diffusion,
    ddim2ddim_reconstruct,
    gather_patches_from_npz,
    visualize_np,
)


def save_side_by_side(
    orig: np.ndarray,
    reco: np.ndarray,
    diff: np.ndarray,
    out_path: Path,
    title: str,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, img, lab, cmap, v0, v1 in [
        (axes[0], orig, "Original", "bwr", vmin, vmax),
        (axes[1], reco, "Reconstructed", "bwr", vmin, vmax),
        (axes[2], diff, "Difference (reco − orig)", "bwr", -0.5, 0.5),
    ]:
        im = ax.imshow(np.squeeze(img), aspect="auto", origin="lower", cmap=cmap, vmin=v0, vmax=v1)
        ax.set_title(lab)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_one(
    npz_path: Path,
    out_dir: Path,
    plot_dir: Path,
    diffusion,
    model,
    *,
    T: int,
    batch_size: int,
    max_patches: int | None,
    label: str,
    heartbeat: Path | None = None,
) -> Path:
    pts, layouts, frame_slices, frame_stems = gather_patches_from_npz(
        npz_path, reco_key="reco", patch_h=512, patch_w=512, scale_div=None
    )
    if max_patches is not None:
        pts = pts[:max_patches]

    pts = visualize_np(np.expand_dims(pts, axis=1)).astype(np.float32)
    n_total = int(pts.shape[0])
    if n_total == 0:
        raise ValueError(f"No patches in {npz_path}")

    ddim_key = f"ddim2ddim-T{T}"
    sal_key = f"saliency-T{T}"
    save_data: dict = {
        "__ddim2ddim_source__": {
            "input_filename": npz_path.name,
            "input_path": str(npz_path.resolve()),
            "input_relative_to_scan_root": None,
            "label": label,
            "layouts": layouts,
            "frame_slices": frame_slices,
            "frame_stems": frame_stems,
            "n_patches": n_total,
            "T": T,
        }
    }

    def _hb() -> None:
        if heartbeat is None:
            return
        heartbeat.parent.mkdir(parents=True, exist_ok=True)
        heartbeat.write_text(f"{npz_path.name}\n")

    dev = dist_util.dev()
    for bstart in range(0, n_total, batch_size):
        _hb()
        bend = min(bstart + batch_size, n_total)
        imgs = th.tensor(pts[bstart:bend], device=dev)
        noised_btch, reco = ddim2ddim_reconstruct(diffusion, model, imgs, T, progress=False)
        for i in range(len(imgs)):
            gidx = bstart + i
            origin = imgs[i].detach().cpu().numpy()
            r = reco[i].detach().cpu().numpy()
            nsd = noised_btch[i].detach().cpu().numpy()
            sal = r - origin
            save_data[gidx] = {
                "original": origin,
                "noised": nsd,
                ddim_key: r,
                sal_key: sal,
            }
            save_side_by_side(
                origin,
                r,
                sal,
                plot_dir / f"{npz_path.stem}_patch{gidx:03d}_T{T}.png",
                title=f"{label} | {npz_path.stem} | patch {gidx} | T={T}",
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / f"{npz_path.stem}_T{T}_ddim2ddim.pkl"
    with pkl_path.open("wb") as fh:
        pickle.dump(save_data, fh, protocol=pickle.HIGHEST_PROTOCOL)
    _hb()
    return pkl_path


def _touch_heartbeat(path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{Path().cwd()}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-root",
        type=Path,
        default=Path("handscan_validation/npz_inference"),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("handscan_validation/inference_T100"),
    )
    p.add_argument(
        "--model-path",
        type=Path,
        default=Path("/exp/sbnd/data/users/gputnam/training-SBND/iterE/results/brats2update111000.pt"),
    )
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="Optional cap on patches per file (e.g. 1 for a quick smoke test)",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        default=["healthy", "unhealthy"],
    )
    p.add_argument("--max-files-per-label", type=int, default=None)
    p.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Process a single NPZ (worker mode). Requires --label.",
    )
    p.add_argument("--label", type=str, default=None, help="Label for --npz worker mode")
    p.add_argument(
        "--heartbeat",
        type=Path,
        default=None,
        help="Touch this file periodically so the orchestrator can detect stale workers",
    )
    args = p.parse_args()

    th.set_grad_enabled(False)
    _touch_heartbeat(args.heartbeat)
    model, diffusion = build_model_and_diffusion()
    sd = dist_util.load_state_dict(str(args.model_path), map_location="cpu")
    model.load_state_dict(sd)
    model.to(dist_util.dev())
    model.eval()
    _touch_heartbeat(args.heartbeat)
    print(f"Loaded {args.model_path} on {dist_util.dev()}", flush=True)

    if args.npz is not None:
        if not args.label:
            raise SystemExit("--label is required with --npz")
        label = args.label
        out_dir = args.output_root / label
        plot_dir = args.output_root / "plots" / label
        pkl = run_one(
            args.npz,
            out_dir,
            plot_dir,
            diffusion,
            model,
            T=args.T,
            batch_size=args.batch_size,
            max_patches=args.max_patches,
            label=label,
            heartbeat=args.heartbeat,
        )
        print(f"wrote {pkl}", flush=True)
        return

    for label in args.labels:
        in_dir = args.input_root / label
        out_dir = args.output_root / label
        plot_dir = args.output_root / "plots" / label
        npzs = sorted(in_dir.glob("*.npz"))
        if args.max_files_per_label is not None:
            npzs = npzs[: args.max_files_per_label]
        print(f"[{label}] {len(npzs)} files → {out_dir}")
        for nz in tqdm(npzs, desc=label):
            pkl = run_one(
                nz,
                out_dir,
                plot_dir,
                diffusion,
                model,
                T=args.T,
                batch_size=args.batch_size,
                max_patches=args.max_patches,
                label=label,
                heartbeat=args.heartbeat,
            )
            print(f"  wrote {pkl.name}", flush=True)


if __name__ == "__main__":
    main()
