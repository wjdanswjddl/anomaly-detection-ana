#!/usr/bin/env python3
"""Submit stop-signal grid inference using *your* synced training checkpoints.

Expects EMA weights under ``--models-root`` (default: durable DATA_ROOT training
copies) and a pnfs file list for the probe patches.

Example::

  python inference/submit_training_ckpts_grid.py --dry-run
  python inference/submit_training_ckpts_grid.py --probe healthy --T 200
  python inference/submit_training_ckpts_grid.py --probe coh --T 100 200
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_USER = os.environ.get("USER", "munjung")
_PNFS = Path(f"/pnfs/sbnd/scratch/users/{_USER}/anomaly-detection")
_DATA = Path(f"/exp/sbnd/data/users/{_USER}/anomaly-detection")

_PROBE_LISTS = {
    "coh": _PNFS / "lists" / "stop_signal_coh_noise_patches.list",
    "healthy": _PNFS / "lists" / "stop_signal_healthy_patches.list",
    "both": _PNFS / "lists" / "stop_signal_healthy_and_coh_noise_patches.list",
}


def _ema_step(path: Path) -> int | None:
    name = path.name
    # emabrats2update_0.9999_010000.pt
    if not name.startswith("emabrats2update_") or not name.endswith(".pt"):
        return None
    try:
        return int(name.rsplit("_", 1)[-1].removesuffix(".pt"))
    except ValueError:
        return None


def pick_ckpts(
    cfg_dir: Path, want_steps: list[int], *, add_newest: bool = True
) -> list[Path]:
    available = {
        s: p
        for p in cfg_dir.glob("emabrats2update_0.9999_*.pt")
        if (s := _ema_step(p)) is not None
    }
    if not available:
        return []
    chosen: list[Path] = []
    for step in want_steps:
        if step in available:
            chosen.append(available[step])
    if add_newest:
        newest = available[max(available)]
        if newest not in chosen:
            chosen.append(newest)
    return sorted(set(chosen), key=lambda p: _ema_step(p) or 0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--configs",
        nargs="+",
        default=["linear", "anisotropic"],
        help="Training configs to submit",
    )
    p.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[10000, 20000, 30000],
        help="EMA steps to submit",
    )
    p.add_argument(
        "--add-newest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also include the newest EMA under each config (default: off)",
    )
    p.add_argument(
        "--models-root",
        type=Path,
        default=_DATA / "training" / "diffusion",
        help="Dir with <config>/emabrats2update_*.pt",
    )
    p.add_argument(
        "--probe",
        choices=sorted(_PROBE_LISTS),
        default="coh",
        help="Which patch list / campaign name prefix to use",
    )
    p.add_argument(
        "--file-list",
        type=Path,
        default=None,
        help="Override file list (default: from --probe)",
    )
    p.add_argument("--T", nargs="+", type=int, default=[100, 200])
    p.add_argument("-ngrid", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    file_list = args.file_list or _PROBE_LISTS[args.probe]
    if not file_list.is_file():
        sys.exit(f"Missing file list: {file_list}")

    env = os.environ.copy()
    env.setdefault("ANOMALY_WD", str(_REPO))
    env.setdefault("ANOMALY_GRID_OUT_DIR", str(_PNFS / "out"))
    env.setdefault("ANOMALY_USE_GIT_CLONE", "0")
    env.setdefault("JOBSUB_MEMORY", "6GB")
    env.setdefault("JOBSUB_DISK", "20GB")
    env.setdefault("JOBSUB_LIFETIME", "12h")
    env.setdefault("JOBSUB_CPU", "4")

    submit_py = _REPO / "inference" / "submit_ddim_grid.py"
    planned: list[tuple[str, Path]] = []
    for cfg in args.configs:
        cfg_dir = args.models_root / cfg
        ckpts = pick_ckpts(cfg_dir, list(args.steps), add_newest=args.add_newest)
        if not ckpts:
            print(f"WARNING: no EMA ckpts under {cfg_dir}", file=sys.stderr)
            continue
        for ck in ckpts:
            planned.append((cfg, ck))

    if not planned:
        sys.exit(
            f"No training checkpoints found under {args.models_root}. "
            "On EAF run: bash bin/sync_eaf_checkpoints.sh"
        )

    print(f"probe={args.probe}  file_list={file_list}")
    print(f"Will submit {len(planned)} campaign(s):")
    for cfg, ck in planned:
        print(f"  {cfg}  step={_ema_step(ck)}  {ck}")

    for cfg, ck in planned:
        step = _ema_step(ck)
        t_tag = "-".join(str(t) for t in args.T)
        # Distinct names so healthy does not displace coh under prefer_campaigns.
        if args.probe == "coh":
            out_name = f"stopsignal_{cfg}_ckpt{step:06d}_T{t_tag}"
        else:
            out_name = f"stopsignal_{args.probe}_{cfg}_ckpt{step:06d}_T{t_tag}"
        cmd = [
            sys.executable,
            str(submit_py),
            "-l",
            str(file_list),
            "--model",
            str(ck),
            "-o",
            out_name,
            "-ngrid",
            str(args.ngrid),
            "--T",
            *[str(t) for t in args.T],
            "--batch-size",
            str(args.batch_size),
            "--extra-args",
            f"--train-config {cfg}",
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        print("\n====", out_name, "====", flush=True)
        subprocess.check_call(cmd, cwd=str(_REPO), env=env)


if __name__ == "__main__":
    main()
