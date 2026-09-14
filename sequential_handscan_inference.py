#!/usr/bin/env python3
"""Single-process handscan inference, alternating healthy / unhealthy.

Faster on CPU than many contended workers. Skips files that already have pickles.
"""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import torch as th

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO / "_stubs"))
sys.path.insert(0, str(_REPO / "train" / "diffusion-anomaly"))

from guided_diffusion import dist_util  # noqa: E402
from run_ddim2ddim_inference import build_model_and_diffusion  # noqa: E402
from run_handscan_validation_inference import run_one  # noqa: E402


def build_alternating_queue(
    input_root: Path,
    output_root: Path,
    T: int,
    *,
    reverse: bool = False,
    start_unhealthy: bool = False,
) -> deque[tuple[str, Path]]:
    by_label: dict[str, list[Path]] = {}
    for label in ("healthy", "unhealthy"):
        pending = []
        done = 0
        for nz in sorted((input_root / label).glob("*.npz")):
            pkl = output_root / label / f"{nz.stem}_T{T}_ddim2ddim.pkl"
            if pkl.is_file():
                done += 1
            else:
                pending.append(nz)
        by_label[label] = pending
        print(f"[{label}] pending={len(pending)} already_done={done}", flush=True)

    order = ("unhealthy", "healthy") if start_unhealthy else ("healthy", "unhealthy")
    q: deque[tuple[str, Path]] = deque()
    while any(by_label.values()):
        for label in order:
            if by_label[label]:
                # pop from end when reverse so we work backwards through sorted names
                nz = by_label[label].pop() if reverse else by_label[label].pop(0)
                q.append((label, nz))
    return q


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-root", type=Path, default=_REPO / "handscan_validation/npz_inference")
    ap.add_argument("--output-root", type=Path, default=_REPO / "handscan_validation/inference_T100")
    ap.add_argument(
        "--model-path",
        type=Path,
        default=Path("/exp/sbnd/data/users/gputnam/training-SBND/iterE/results/brats2update111000.pt"),
    )
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-patches", type=int, default=None)
    ap.add_argument(
        "--reverse",
        action="store_true",
        help="Work backwards through sorted filenames within each label",
    )
    ap.add_argument(
        "--start-unhealthy",
        action="store_true",
        help="Interleave starting with unhealthy (default starts with healthy)",
    )
    args = ap.parse_args()

    queue = build_alternating_queue(
        args.input_root,
        args.output_root,
        args.T,
        reverse=args.reverse,
        start_unhealthy=args.start_unhealthy,
    )
    print(
        f"Single worker queue={len(queue)} reverse={args.reverse} "
        f"start_unhealthy={args.start_unhealthy} on {dist_util.dev()}",
        flush=True,
    )
    if queue:
        print(f"  first: {queue[0][0]}/{queue[0][1].name}", flush=True)
        print(f"  last:  {queue[-1][0]}/{queue[-1][1].name}", flush=True)

    th.set_grad_enabled(False)
    model, diffusion = build_model_and_diffusion()
    sd = dist_util.load_state_dict(str(args.model_path), map_location="cpu")
    model.load_state_dict(sd)
    model.to(dist_util.dev())
    model.eval()
    print(f"Loaded {args.model_path.name}", flush=True)

    for i, (label, nz) in enumerate(queue, 1):
        out_dir = args.output_root / label
        plot_dir = args.output_root / "plots" / label
        pkl = out_dir / f"{nz.stem}_T{args.T}_ddim2ddim.pkl"
        if pkl.is_file():
            print(f"[{i}/{len(queue)}] skip {label}/{nz.name} (exists)", flush=True)
            continue
        print(f"[{i}/{len(queue)}] {label}/{nz.name} …", flush=True)
        out = run_one(
            nz,
            out_dir,
            plot_dir,
            diffusion,
            model,
            T=args.T,
            batch_size=args.batch_size,
            max_patches=args.max_patches,
            label=label,
            heartbeat=None,
        )
        print(f"  wrote {out.name}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
