#!/usr/bin/env python3
"""Build handscan NPZ datasets for grid inference from archived notes.

Splits:
  healthy  — note ``clean plane 1``, plane-1 crop wires [3000, 3968)
  diseased — note ``streaks on plane 1``, same crop
  outlier  — ``strong coherent noise``, ``high energy shower``, and
             ``high energy shower, steps on plane 0, steps on plane 1`` at full plane range

Writes inference NPZs under DATA_ROOT and copies to pnfs for jobsub file lists.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from configs.paths import DATA_ROOT, HANDSCAN_ARCHIVES, ensure_layout  # noqa: E402

PLANE_EDGES = [0, 1984, 3968, 5638]
PLANE_SCALES = {0: 200.0, 1: 100.0, 2: 200.0}
PATCH = 512

HEALTHY_NOTE = "clean plane 1"
DISEASED_NOTE = "streaks on plane 1"
OUTLIER_NOTES = {
    "strong coherent noise": [1],
    "high energy shower": [1],
    "high energy shower, steps on plane 0, steps on plane 1": [0, 1],
}

CROP_LO, CROP_HI = 3000, 3968
# Legacy auto-grid paths (relocated). Prefer curated patches from CurateHandscanPatches.
_PNFS_INPUTS = Path(
    "/pnfs/sbnd/scratch/users/munjung/anomaly-detection/inputs/_archived_handscan_grid_auto/handscan_grid"
)
_PNFS_LISTS = Path("/pnfs/sbnd/scratch/users/munjung/anomaly-detection/lists/_archived")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dedupe_by_sample_id(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for r in rows:
        sid = r["sample_id"]
        if sid in seen:
            continue
        seen.add(sid)
        out.append(r)
    return out


def safe_stem(sample_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id)


def load_waveform(record: dict) -> np.ndarray:
    with h5py.File(record["h5_path"], "r") as hf:
        return np.asarray(hf[record["dataset_path"]], dtype=np.float32)


def crop_wires(arr: np.ndarray, lo: int, hi: int) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D waveform, got {arr.shape}")
    if hi > arr.shape[0]:
        raise ValueError(f"Wire hi={hi} exceeds n_wires={arr.shape[0]} for {arr.shape}")
    return arr[lo:hi, :].astype(np.float32, copy=False)


def plane_slice(plane_idx: int) -> tuple[int, int]:
    return PLANE_EDGES[plane_idx], PLANE_EDGES[plane_idx + 1]


def make_inference_plane(crop: np.ndarray, scale: float, patch: int = PATCH) -> np.ndarray:
    plane = crop.astype(np.float32) / float(scale)
    h, w = plane.shape
    if h > patch:
        plane = plane[:patch, :]
    elif h < patch:
        pad = patch - h
        plane = np.pad(plane, ((0, pad), (0, 0)), mode="reflect")
    wc = (w // patch) * patch
    if wc < patch:
        raise ValueError(f"Tick dim {w} smaller than patch {patch}")
    return plane[:, :wc].astype(np.float32, copy=False)


def build_reco_stack(
    waveform: np.ndarray,
    *,
    wire_lo: int,
    wire_hi: int,
    plane_indices: list[int] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Return reco (N, H, W) and per-frame metadata."""
    frames: list[np.ndarray] = []
    meta_frames: list[dict] = []
    if plane_indices is None:
        crop = crop_wires(waveform, wire_lo, wire_hi)
        scale = PLANE_SCALES[1]
        plane = make_inference_plane(crop, scale)
        frames.append(plane)
        meta_frames.append(
            {
                "wire_lo": wire_lo,
                "wire_hi": wire_hi,
                "plane_idx": 1,
                "scale": scale,
                "crop_shape": list(crop.shape),
                "inference_shape": list(plane.shape),
            }
        )
    else:
        for pidx in plane_indices:
            lo, hi = plane_slice(pidx)
            crop = crop_wires(waveform, lo, hi)
            scale = PLANE_SCALES[pidx]
            plane = make_inference_plane(crop, scale)
            frames.append(plane)
            meta_frames.append(
                {
                    "wire_lo": lo,
                    "wire_hi": hi,
                    "plane_idx": pidx,
                    "scale": scale,
                    "crop_shape": list(crop.shape),
                    "inference_shape": list(plane.shape),
                }
            )
    stack = np.stack(frames, axis=0).astype(np.float32, copy=False)
    return stack, meta_frames


def write_npz(
    path: Path,
    reco: np.ndarray,
    record: dict,
    *,
    label: str,
    wire_lo: int,
    wire_hi: int,
    scale: float,
    meta_frames: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        reco=reco,
        wire_lo=np.int32(wire_lo),
        wire_hi=np.int32(wire_hi),
        scale=np.float32(scale),
        sample_id=np.asarray(record["sample_id"]),
        note=np.asarray(record.get("note", "")),
        label=np.asarray(label),
        meta_frames=np.asarray(json.dumps(meta_frames)),
    )


def process_records(
    records: list[dict],
    label: str,
    out_dir: Path,
    *,
    wire_lo: int,
    wire_hi: int,
    plane_indices: list[int] | None = None,
) -> list[dict]:
    manifest: list[dict] = []
    for rec in records:
        stem = safe_stem(rec["sample_id"])
        npz_path = out_dir / f"{stem}.npz"
        waveform = load_waveform(rec)
        reco, meta_frames = build_reco_stack(
            waveform,
            wire_lo=wire_lo,
            wire_hi=wire_hi,
            plane_indices=plane_indices,
        )
        primary_scale = meta_frames[0]["scale"]
        write_npz(
            npz_path,
            reco,
            rec,
            label=label,
            wire_lo=wire_lo,
            wire_hi=wire_hi,
            scale=primary_scale,
            meta_frames=meta_frames,
        )
        manifest.append(
            {
                **rec,
                "label": label,
                "stem": stem,
                "inference_npz": str(npz_path.resolve()),
                "n_frames": int(reco.shape[0]),
                "reco_shape": list(reco.shape),
                "wire_lo": wire_lo,
                "wire_hi": wire_hi,
                "meta_frames": meta_frames,
            }
        )
    return manifest


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def write_list(path: Path, npz_paths: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(p.resolve()) for p in sorted(npz_paths)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def copy_to_pnfs(src_dir: Path, pnfs_dir: Path) -> list[Path]:
    pnfs_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src in sorted(src_dir.glob("*.npz")):
        dest = pnfs_dir / src.name
        if not dest.exists() or dest.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dest)
        copied.append(dest)
    return copied


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--archive",
        type=Path,
        default=HANDSCAN_ARCHIVES / "offbeamlight_v10_06_00" / "unhealthy.jsonl",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=DATA_ROOT / "archive" / "handscan_grid_auto_uncurated" / "grid",
        help="Legacy auto full-plane crops (archived). Prefer CurateHandscanPatches.ipynb.",
    )
    p.add_argument("--skip-pnfs", action="store_true")
    args = p.parse_args()

    ensure_layout()
    out_root = args.out_root
    npz_root = out_root / "npz_inference"
    manifest_dir = out_root / "manifests"

    rows = load_jsonl(args.archive)
    healthy = dedupe_by_sample_id([r for r in rows if r.get("note", "").strip() == HEALTHY_NOTE])
    diseased = dedupe_by_sample_id([r for r in rows if r.get("note", "").strip() == DISEASED_NOTE])
    outlier_records: list[dict] = []
    for note in OUTLIER_NOTES:
        outlier_records.extend(
            dedupe_by_sample_id([r for r in rows if r.get("note", "").strip() == note])
        )
    outlier_records = dedupe_by_sample_id(outlier_records)

    print(f"Archive rows: {len(rows)}")
    print(f"  healthy : {len(healthy)}  ({HEALTHY_NOTE})")
    print(f"  diseased: {len(diseased)}  ({DISEASED_NOTE})")
    print(f"  outlier : {len(outlier_records)}")

    healthy_manifest = process_records(
        healthy,
        "healthy",
        npz_root / "healthy",
        wire_lo=CROP_LO,
        wire_hi=CROP_HI,
    )
    diseased_manifest = process_records(
        diseased,
        "diseased",
        npz_root / "diseased",
        wire_lo=CROP_LO,
        wire_hi=CROP_HI,
    )

    outlier_manifest: list[dict] = []
    for rec in outlier_records:
        note = rec.get("note", "").strip()
        plane_indices = OUTLIER_NOTES[note]
        lo, hi = plane_slice(plane_indices[0])
        if len(plane_indices) == 1:
            wire_lo, wire_hi = lo, hi
        else:
            wire_lo, wire_hi = plane_slice(plane_indices[0])[0], plane_slice(plane_indices[-1])[1]
        one = process_records(
            [rec],
            "outlier",
            npz_root / "outlier",
            wire_lo=wire_lo,
            wire_hi=wire_hi,
            plane_indices=plane_indices if len(plane_indices) > 1 else None,
        )
        outlier_manifest.extend(one)

    write_manifest(manifest_dir / "healthy.jsonl", healthy_manifest)
    write_manifest(manifest_dir / "diseased.jsonl", diseased_manifest)
    write_manifest(manifest_dir / "outlier.jsonl", outlier_manifest)

    summary = {
        "healthy_n": len(healthy_manifest),
        "diseased_n": len(diseased_manifest),
        "outlier_n": len(outlier_manifest),
        "crop_lo": CROP_LO,
        "crop_hi": CROP_HI,
        "plane_edges": PLANE_EDGES,
        "outlier_notes": list(OUTLIER_NOTES.keys()),
    }
    (manifest_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    if args.skip_pnfs:
        return

    pnfs_by_label = {
        "healthy": copy_to_pnfs(npz_root / "healthy", _PNFS_INPUTS / "healthy"),
        "diseased": copy_to_pnfs(npz_root / "diseased", _PNFS_INPUTS / "diseased"),
        "outlier": copy_to_pnfs(npz_root / "outlier", _PNFS_INPUTS / "outlier"),
    }
    all_paths = pnfs_by_label["healthy"] + pnfs_by_label["diseased"] + pnfs_by_label["outlier"]
    write_list(_PNFS_LISTS / "handscan_grid_healthy.list", pnfs_by_label["healthy"])
    write_list(_PNFS_LISTS / "handscan_grid_diseased.list", pnfs_by_label["diseased"])
    write_list(_PNFS_LISTS / "handscan_grid_outlier.list", pnfs_by_label["outlier"])
    write_list(_PNFS_LISTS / "handscan_grid_all.list", all_paths)
    print(f"Wrote pnfs lists under {_PNFS_LISTS}")
    print(f"  all: {len(all_paths)} NPZs")


if __name__ == "__main__":
    main()
