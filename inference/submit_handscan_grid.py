#!/usr/bin/env python3
"""Submit handscan grid inference for curated patch NPZs.

Default file list is ``handscan_curated_all.list`` (from
``samples/CurateHandscanPatches.ipynb``). Auto-generated full-plane crops
were relocated under ``lists/_archived/`` and should not be submitted.

Uses latest synced training EMA checkpoints (linear + anisotropic).

Example::

  python inference/submit_handscan_grid.py --dry-run
  python inference/submit_handscan_grid.py --list handscan_curated_healthy.list --modes ddim2ddim --T 100 200
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

_ALL_MODES = ["ddim2ddim", "rand2ddim", "rand2ddpm", "ddpm2ddpm", "ddim2ddpm"]
_DEFAULT_TS = list(range(100, 801, 100))


def _ema_step(path: Path) -> int | None:
    name = path.name
    if not name.startswith("emabrats2update_") or not name.endswith(".pt"):
        return None
    try:
        return int(name.rsplit("_", 1)[-1].removesuffix(".pt"))
    except ValueError:
        return None


def pick_newest_ckpt(cfg_dir: Path) -> Path | None:
    available = {
        s: p
        for p in cfg_dir.glob("emabrats2update_0.9999_*.pt")
        if (s := _ema_step(p)) is not None
    }
    if not available:
        return None
    return available[max(available)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--list",
        default="handscan_curated_all.list",
        help="File list basename under pnfs lists/ (or absolute path)",
    )
    p.add_argument(
        "--configs",
        nargs="+",
        default=["linear", "anisotropic"],
        help="Training configs to submit",
    )
    p.add_argument(
        "--models-root",
        type=Path,
        default=_DATA / "training" / "diffusion",
    )
    p.add_argument("--T", nargs="+", type=int, default=_DEFAULT_TS)
    p.add_argument("--modes", nargs="+", default=_ALL_MODES)
    p.add_argument(
        "--output-prefix",
        default="handscan_curated",
        help="Campaign name prefix",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    file_list = Path(args.list)
    if not file_list.is_file():
        file_list = _PNFS / "lists" / args.list
    if not file_list.is_file():
        archived = _PNFS / "lists" / "_archived" / Path(args.list).name
        hint = ""
        if archived.is_file():
            hint = (
                f" Found archived auto-grid list at {archived} — "
                "use curated lists from CurateHandscanPatches.ipynb instead."
            )
        sys.exit(
            f"Missing file list: {file_list}.{hint} "
            "Build with samples/CurateHandscanPatches.ipynb (stage to pnfs cell)."
        )

    n_inputs = sum(
        1
        for line in file_list.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    )
    if n_inputs == 0:
        sys.exit(f"No inputs in {file_list}")

    env = os.environ.copy()
    env.setdefault("ANOMALY_WD", str(_REPO))
    env.setdefault("ANOMALY_GRID_OUT_DIR", str(_PNFS / "out"))
    env.setdefault("ANOMALY_USE_GIT_CLONE", "0")
    env.setdefault("JOBSUB_MEMORY", "8GB")
    env.setdefault("JOBSUB_DISK", "40GB")
    env.setdefault("JOBSUB_LIFETIME", "48h")
    env.setdefault("JOBSUB_CPU", "4")

    submit_py = _REPO / "inference" / "submit_ddim_grid.py"
    planned: list[tuple[str, Path, int]] = []
    for cfg in args.configs:
        cfg_dir = args.models_root / cfg
        ckpt = pick_newest_ckpt(cfg_dir)
        if ckpt is None:
            print(f"WARNING: no EMA ckpts under {cfg_dir}", file=sys.stderr)
            continue
        planned.append((cfg, ckpt, _ema_step(ckpt) or 0))

    if not planned:
        sys.exit(
            f"No training checkpoints found under {args.models_root}. "
            "On EAF run: bash bin/sync_eaf_checkpoints.sh"
        )

    t_tag = "-".join(str(t) for t in args.T)
    mode_tag = "-".join(args.modes)
    print(f"file_list={file_list}  n_inputs={n_inputs}")
    print(f"T={args.T}  modes={args.modes}")
    print(f"Will submit {len(planned)} campaign(s):")
    for cfg, ck, step in planned:
        print(f"  {cfg}  step={step}  {ck}")

    for cfg, ck, step in planned:
        out_name = f"{args.output_prefix}_{cfg}_ckpt{step:06d}_T{t_tag}_{mode_tag}"
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
            str(n_inputs),
            "--T",
            *[str(t) for t in args.T],
            "--batch-size",
            "1",
            "--modes",
            *args.modes,
            "--extra-args",
            f"--train-config {cfg}",
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        print("\n====", out_name, "====", flush=True)
        subprocess.check_call(cmd, cwd=str(_REPO), env=env)


if __name__ == "__main__":
    main()
