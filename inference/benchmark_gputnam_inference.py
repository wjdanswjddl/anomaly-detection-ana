#!/usr/bin/env python3
"""Benchmark Gray Putnam iterE DDIM→DDIM inference on CPU and/or GPU.

Uses the production EMA checkpoint and the same reconstruction path as
``run_ddim2ddim_inference.py`` (T noise encode + T denoise).

Examples::

  # CPU only (sbndbuild / no CUDA)
  python inference/benchmark_gputnam_inference.py --device cpu --T 200

  # GPU (EAF diffusion kernel)
  python inference/benchmark_gputnam_inference.py --device cuda --T 200

  # Both when CUDA is visible
  python inference/benchmark_gputnam_inference.py --device auto --T 200 --T 100
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch as th

_REPO = Path(__file__).resolve().parents[1]
_STUBS = _REPO / "_stubs"
_DIFFUSION = _REPO / "train" / "diffusion-anomaly"
sys.path.insert(0, str(_STUBS))
sys.path.insert(0, str(_DIFFUSION))
sys.path.insert(0, str(_REPO))

from inference.run_ddim2ddim_inference import (  # noqa: E402
    build_model_and_diffusion,
    ddim2ddim_reconstruct,
    patches_from_plane,
    visualize_np,
)

DEFAULT_MODEL = Path(
    "/exp/sbnd/data/users/gputnam/training-SBND/iterE/results/"
    "emabrats2update_0.9999_111000.pt"
)
DEFAULT_NPZ = Path(
    "/exp/sbnd/data/users/munjung/anomaly-detection/samples/handscan/"
    "validation/npz_inference/healthy/86414745_63_g4-raw-0_11.h5_8.npz"
)
DEFAULT_OUT = Path(
    "/exp/sbnd/data/users/munjung/anomaly-detection/training/benchmarks"
)


def _pick_devices(want: str) -> list[str]:
    if want == "cpu":
        return ["cpu"]
    if want == "cuda":
        if not th.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
        return ["cuda"]
    # auto
    out = ["cpu"]
    if th.cuda.is_available():
        out.append("cuda")
    return out


def _load_patches(npz_path: Path, n_patches: int, patch: int) -> np.ndarray:
    z = np.load(npz_path)
    reco = z["reco"]
    if reco.ndim == 3 and reco.shape[0] == 1:
        plane = reco[0]
    elif reco.ndim == 2:
        plane = reco
    elif reco.ndim == 4:
        plane = reco[0, 0]
    else:
        raise ValueError(f"Unexpected reco shape {reco.shape}")
    plane = visualize_np(plane.astype(np.float32))
    patches, _ = patches_from_plane(plane, patch, patch)
    if n_patches > 0:
        patches = patches[:n_patches]
    return patches


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        th.cuda.synchronize()


def _time_once(diffusion, model, batch: th.Tensor, T: int) -> float:
    _sync(str(batch.device))
    t0 = time.perf_counter()
    with th.no_grad():
        _, reco = ddim2ddim_reconstruct(diffusion, model, batch, T, progress=False)
    _sync(str(batch.device))
    dt = time.perf_counter() - t0
    # Touch result so nothing is optimized away
    _ = float(reco.detach().float().mean().cpu())
    return dt


def run_device(
    *,
    device: str,
    model_path: Path,
    patches: np.ndarray,
    T: int,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> dict:
    th.set_grad_enabled(False)

    model, diffusion = build_model_and_diffusion(
        {"noise_schedule": "linear", "predict_xstart": False, "anisotropic_noise": False}
    )
    try:
        sd = th.load(str(model_path), map_location="cpu", weights_only=False)
    except TypeError:
        sd = th.load(str(model_path), map_location="cpu")
    model.load_state_dict(sd)
    model.to(th.device(device))
    model.eval()

    # Take first batch_size patches (or fewer)
    n = min(batch_size, patches.shape[0])
    x = th.from_numpy(patches[:n]).float().unsqueeze(1)  # (N,1,H,W)
    x = x.to(th.device(device))

    info = {
        "device": device,
        "torch_device": str(x.device),
        "T": T,
        "batch_size": int(n),
        "patch_shape": list(patches.shape[1:]),
        "model_path": str(model_path),
        "ddim_steps_total": 2 * T,  # encode + decode
    }
    if device.startswith("cuda") and th.cuda.is_available():
        info["gpu_name"] = th.cuda.get_device_name(0)
        info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")

    for i in range(warmup):
        dt = _time_once(diffusion, model, x, T)
        print(f"  [{device}] warmup {i+1}/{warmup}: {dt:.3f}s", flush=True)

    times = []
    for i in range(repeats):
        dt = _time_once(diffusion, model, x, T)
        times.append(dt)
        print(
            f"  [{device}] timed {i+1}/{repeats}: {dt:.3f}s "
            f"({dt/n:.3f}s/patch, {dt/(2*T):.4f}s/DDIM-step)",
            flush=True,
        )

    info["times_s"] = times
    info["mean_s"] = statistics.mean(times)
    info["stdev_s"] = statistics.stdev(times) if len(times) > 1 else 0.0
    info["mean_s_per_patch"] = info["mean_s"] / n
    info["mean_s_per_ddim_step"] = info["mean_s"] / (2 * T)
    info["patches_per_hour"] = 3600.0 / info["mean_s_per_patch"]
    return info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--T", type=int, action="append", default=None, help="Repeatable; default 200")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--n-patches", type=int, default=1, help="Patches loaded from NPZ (before batch)")
    p.add_argument("--patch-size", type=int, default=512)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    Ts = args.T or [200]
    if not args.model_path.is_file():
        sys.exit(f"Missing model: {args.model_path}")
    if not args.npz.is_file():
        sys.exit(f"Missing npz: {args.npz}")

    devices = _pick_devices(args.device)
    patches = _load_patches(args.npz, args.n_patches, args.patch_size)
    print(f"host={platform.node()}")
    print(f"torch={th.__version__} cuda_available={th.cuda.is_available()}")
    print(f"model={args.model_path}")
    print(f"npz={args.npz} -> patches {patches.shape}")
    print(f"devices={devices} Ts={Ts} batch={args.batch_size} warmup={args.warmup} repeats={args.repeats}")

    report = {
        "host": platform.node(),
        "torch_version": th.__version__,
        "cuda_available": bool(th.cuda.is_available()),
        "model_path": str(args.model_path),
        "npz": str(args.npz),
        "runs": [],
    }

    for T in Ts:
        for device in devices:
            print(f"\n=== device={device} T={T} ===", flush=True)
            # Fresh process-like isolation for CPU after CUDA is awkward;
            # run CPU first in auto mode (devices list is cpu then cuda).
            run = run_device(
                device=device,
                model_path=args.model_path,
                patches=patches,
                T=T,
                batch_size=args.batch_size,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            report["runs"].append(run)
            print(
                f"→ {device} T={T}: mean={run['mean_s']:.3f}s ± {run['stdev_s']:.3f}s | "
                f"{run['mean_s_per_patch']:.3f}s/patch | "
                f"{run['patches_per_hour']:.1f} patches/hour",
                flush=True,
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = args.out_dir / f"gputnam_iterE_inference_bench_{stamp}.json"
    out.write_text(json.dumps(report, indent=2))
    latest = args.out_dir / "gputnam_iterE_inference_bench_latest.json"
    latest.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")
    print(f"Wrote {latest}")

    # Compact table
    print("\nSummary")
    print(f"{'device':8} {'T':>5} {'mean_s':>10} {'s/patch':>10} {'patches/h':>10}")
    for r in report["runs"]:
        print(
            f"{r['device']:8} {r['T']:5d} {r['mean_s']:10.3f} "
            f"{r['mean_s_per_patch']:10.3f} {r['patches_per_hour']:10.1f}"
        )


if __name__ == "__main__":
    main()
