#!/usr/bin/env python3
"""Build SBND ROC probe: 200 healthy 512² MC patches + 3 injected defect sets.

Healthy patches come from the **MC training distribution** (not handscan):

1. Prefer ``DATA_DIR/healthy/*.npz`` — the same tree used to train the diffusion
   models (``/scratch/.../npz/healthy`` on EAF, or a durable sync under
   ``samples/npz/healthy``).
2. Else tile SBND raw ``samples/h5/*.h5`` with the same plane scales / 512 tiling
   as ``guided_diffusion.image_datasets.load_image_file``.

Injects bad_wire / coh_noise / charge_tail and stages pnfs inputs + file lists
(200 files → ``-ngrid 50``).

Example::

  python samples/build_sbnd_roc_defect_dataset.py
  python samples/build_sbnd_roc_defect_dataset.py --n-patches 200 --seed 42
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "train" / "diffusion-anomaly"))

from configs.paths import (  # noqa: E402
    SAMPLES_DEFECTS,
    SAMPLES_H5,
    SAMPLES_NPZ,
    SCRATCH_NPZ,
    ensure_layout,
    resolve_training_data_dir,
)

PNFS = Path("/pnfs/sbnd/scratch/users/munjung/anomaly-detection")
PNFS_INPUTS = PNFS / "inputs" / "roc_sbnd_defects"
PNFS_LISTS = PNFS / "lists"

DEFECTS = ("bad_wire", "coh_noise", "charge_tail")
PATCH = 512


def _discover_healthy_dirs() -> list[Path]:
    """Ordered candidate dirs of training-style healthy MC NPZs."""
    out: list[Path] = []
    try:
        data_dir = resolve_training_data_dir()
        out.append(data_dir / "healthy")
        out.append(data_dir)
    except FileNotFoundError:
        pass
    if SCRATCH_NPZ is not None:
        out.append(SCRATCH_NPZ / "healthy")
        out.append(SCRATCH_NPZ)
    out.append(SAMPLES_NPZ / "healthy")
    out.append(SAMPLES_NPZ)
    # de-dupe while preserving order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        rp = p.resolve() if p.exists() else p
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(p)
    return uniq


def _list_training_npz(healthy_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for d in healthy_dirs:
        if not d.is_dir():
            continue
        found = sorted(d.glob("*.npz"))
        # Prefer names that look like Gray/training MC patches
        mc = [p for p in found if "tpc" in p.name and "_rec_" in p.name]
        files.extend(mc if mc else found)
        if files:
            break
    return files


def _reco_frames(src: Path) -> np.ndarray:
    """Load reco as float32 (N, 1, H, W)."""
    z = np.load(src, allow_pickle=True)
    if "reco" not in z.files:
        raise ValueError(f"No 'reco' key in {src}")
    reco = np.asarray(z["reco"], dtype=np.float32)
    if reco.ndim == 2:
        reco = reco[np.newaxis, np.newaxis, ...]
    elif reco.ndim == 3:
        if reco.shape[0] == 1:
            reco = reco[np.newaxis, ...]
        else:
            reco = reco[:, np.newaxis, ...]
    elif reco.ndim == 4:
        if reco.shape[1] != 1:
            reco = reco[:, :1]
    else:
        raise ValueError(f"Unexpected reco shape {reco.shape} in {src}")
    if reco.shape[-2:] != (PATCH, PATCH):
        raise ValueError(f"Expected {PATCH}x{PATCH}, got {reco.shape} in {src}")
    return reco


def collect_from_training_npz(
    out_dir: Path,
    *,
    n_needed: int,
    seed: int,
) -> tuple[list[Path], str]:
    """Sample single-frame NPZs from training healthy MC NPZs."""
    files = _list_training_npz(_discover_healthy_dirs())
    if not files:
        return [], ""

    rng = np.random.default_rng(seed)
    # Build (file_idx, frame_idx) candidates; skip corrupt files
    candidates: list[tuple[Path, int]] = []
    src_root = ""
    for src in files:
        try:
            reco = _reco_frames(src)
        except Exception as exc:  # noqa: BLE001 — skip bad training files
            print(f"  skip {src.name}: {exc}", flush=True)
            continue
        if not src_root:
            src_root = str(src.parent)
        for fi in range(reco.shape[0]):
            candidates.append((src, fi))

    if len(candidates) < n_needed:
        print(
            f"  warning: only {len(candidates)} frames in training NPZs "
            f"(need {n_needed})",
            flush=True,
        )
    order = rng.permutation(len(candidates))
    written: list[Path] = []
    for j in order:
        if len(written) >= n_needed:
            break
        src, fi = candidates[int(j)]
        reco = _reco_frames(src)
        frame = reco[fi : fi + 1]  # (1,1,H,W)
        i = len(written)
        # Avoid apply_detector skip substrings tpc1-plane1 / tpc2-plane1
        safe = src.stem.replace("plane1", "p1").replace("-", "_")
        dest = out_dir / f"healthy_{i:04d}_{safe}_f{fi}.npz"
        np.savez_compressed(
            dest,
            reco=frame.astype(np.float32),
            label=np.asarray("healthy"),
            source_file=np.asarray(str(src)),
            frame=np.int32(fi),
            origin=np.asarray("training_mc_npz"),
        )
        written.append(dest)
    return written, src_root


def collect_from_sbnd_h5(
    out_dir: Path,
    *,
    n_needed: int,
    seed: int,
    h5_dir: Path,
) -> tuple[list[Path], str]:
    """Tile SBND raw H5 with the same preprocessing as training ImageDataset."""
    from guided_diffusion.image_datasets import load_image_file  # noqa: WPS

    h5s = sorted(h5_dir.glob("*.h5"))
    if not h5s:
        return [], ""

    rng = np.random.default_rng(seed)
    candidates: list[tuple[str, np.ndarray]] = []
    for h5 in h5s:
        try:
            images, _truth = load_image_file(str(h5), PATCH)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {h5.name}: {exc}", flush=True)
            continue
        # images: (N,1,H,W) already /plane_scale and clipped to [-1,1]
        for fi in range(images.shape[0]):
            candidates.append((h5.name, images[fi : fi + 1]))

    if not candidates:
        return [], ""
    order = rng.permutation(len(candidates))
    written: list[Path] = []
    for j in order:
        if len(written) >= n_needed:
            break
        h5_name, frame = candidates[int(j)]
        i = len(written)
        stem = Path(h5_name).stem.replace("-", "_")
        dest = out_dir / f"healthy_{i:04d}_{stem}_tile{j}.npz"
        np.savez_compressed(
            dest,
            reco=frame.astype(np.float32),
            label=np.asarray("healthy"),
            source_file=np.asarray(str(h5_dir / h5_name)),
            origin=np.asarray("sbnd_h5_tiled"),
        )
        written.append(dest)
    return written, str(h5_dir)


def inject_defects(healthy_dir: Path, defects: tuple[str, ...], seed: int) -> None:
    candidates = [
        Path("/exp/sbnd/app/users/munjung/env/bin/python"),
        Path(sys.executable),
    ]
    py = next((c for c in candidates if c.is_file()), Path(sys.executable))
    cmd = [
        str(py),
        str(_REPO / "samples" / "apply_detector_defects_npz.py"),
        str(healthy_dir),
        "--defects",
        *defects,
        "--seed",
        str(seed),
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(_REPO))


def stage_class(local_dir: Path, class_name: str) -> Path:
    dest = PNFS_INPUTS / f"{class_name}_patches"
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*.npz"):
        old.unlink()
    paths: list[Path] = []
    for src in sorted(local_dir.glob("*.npz")):
        d = dest / src.name
        shutil.copy2(src, d)
        paths.append(d)
    list_path = PNFS_LISTS / f"roc_sbnd_{class_name}_200.list"
    PNFS_LISTS.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(str(p) for p in paths) + ("\n" if paths else ""))
    print(f"  {class_name:12s}  n={len(paths):3d}  → {list_path}")
    return list_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-patches", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-root",
        type=Path,
        default=SAMPLES_DEFECTS / "roc_sbnd_200",
    )
    p.add_argument(
        "--h5-dir",
        type=Path,
        default=SAMPLES_H5,
        help="Fallback SBND raw H5 dir when training NPZs are unavailable",
    )
    p.add_argument("--skip-inject", action="store_true")
    p.add_argument("--skip-stage", action="store_true")
    args = p.parse_args()

    ensure_layout()
    out_root = args.out_root
    if out_root.exists():
        shutil.rmtree(out_root)
    healthy_dir = out_root / "healthy_patches"
    healthy_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building {args.n_patches} healthy MC patches → {healthy_dir}")
    written, src = collect_from_training_npz(
        healthy_dir, n_needed=args.n_patches, seed=args.seed
    )
    origin = "training_mc_npz"
    if written:
        print(f"  from training NPZs ({src}): {len(written)}")
    else:
        print("  no training NPZs on this host — tiling SBND raw H5…")
        written, src = collect_from_sbnd_h5(
            healthy_dir,
            n_needed=args.n_patches,
            seed=args.seed,
            h5_dir=args.h5_dir,
        )
        origin = "sbnd_h5_tiled"
        print(f"  from H5 ({src}): {len(written)}")

    if len(written) < args.n_patches:
        sys.exit(
            f"Only collected {len(written)}/{args.n_patches} healthy patches. "
            "Sync training NPZs from EAF (bin/sync_eaf_training_npz.sh) "
            "or provide --h5-dir with SBND raw MC."
        )

    summary = {
        "n_healthy": len(written),
        "seed": args.seed,
        "healthy_dir": str(healthy_dir),
        "defects": list(DEFECTS),
        "origin": origin,
        "source": src,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    if not args.skip_inject:
        for dname in list(out_root.iterdir()):
            if dname.is_dir() and dname.name.startswith("healthy_patches_"):
                shutil.rmtree(dname)
        inject_defects(healthy_dir, DEFECTS, args.seed)

    if args.skip_stage:
        return

    print("Staging to pnfs…")
    stage_class(healthy_dir, "healthy")
    for defect in DEFECTS:
        defect_dir = out_root / f"healthy_patches_{defect}"
        if not defect_dir.is_dir():
            sys.exit(f"Missing defect dir {defect_dir} — inject first")
        stage_class(defect_dir, defect)

    print("Done. Submit with:")
    print("  bash bin/submit_sbnd_roc_defects.sh")


if __name__ == "__main__":
    main()
