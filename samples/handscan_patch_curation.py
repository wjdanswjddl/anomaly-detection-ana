#!/usr/bin/env python3
"""Helpers for interactive handscan patch curation → job-ready NPZs.

Used by ``samples/CurateHandscanPatches.ipynb``.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import h5py
import numpy as np

PLANE_EDGES = [0, 1984, 3968, 5638]
PLANE_SCALES = {0: 200.0, 1: 100.0, 2: 200.0}
PATCH = 512

# Note-class → curation label
CLASS_SPECS: dict[str, dict] = {
    "healthy": {
        "notes": ["clean plane 1"],
        "default_plane": 1,
        "default_wire_lo": 3000,
        "default_wire_hi": 3968,
    },
    "unhealthy": {
        "notes": ["streaks on plane 1"],
        "default_plane": 1,
        "default_wire_lo": 3000,
        "default_wire_hi": 3968,
    },
    "unique": {
        "notes": [
            "strong coherent noise",
            "high energy shower",
            "high energy shower, steps on plane 0, steps on plane 1",
        ],
        "default_plane": 1,
        "default_wire_lo": None,  # full plane
        "default_wire_hi": None,
    },
}

PNFS_INPUTS = Path("/pnfs/sbnd/scratch/users/munjung/anomaly-detection/inputs/handscan_curated")
PNFS_LISTS = Path("/pnfs/sbnd/scratch/users/munjung/anomaly-detection/lists")


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


def load_records_for_class(archive_jsonl: Path, class_name: str) -> list[dict]:
    if class_name not in CLASS_SPECS:
        raise KeyError(f"Unknown class {class_name!r}; choose from {sorted(CLASS_SPECS)}")
    notes = {n.strip().lower() for n in CLASS_SPECS[class_name]["notes"]}
    rows = load_jsonl(archive_jsonl)
    matched = [r for r in rows if r.get("note", "").strip().lower() in notes]
    return dedupe_by_sample_id(matched)


def load_waveform(record: dict) -> np.ndarray:
    with h5py.File(record["h5_path"], "r") as hf:
        return np.asarray(hf[record["dataset_path"]], dtype=np.float32)


def plane_bounds(plane_idx: int) -> tuple[int, int]:
    return PLANE_EDGES[plane_idx], PLANE_EDGES[plane_idx + 1]


def clamp_crop_to_plane(
    plane_idx: int, wire_lo: int, wire_hi: int
) -> tuple[int, int]:
    plo, phi = plane_bounds(plane_idx)
    lo = max(plo, min(wire_lo, phi - 1))
    hi = max(lo + 1, min(wire_hi, phi))
    return lo, hi


def extract_crop(
    waveform: np.ndarray,
    *,
    wire_lo: int,
    wire_hi: int,
    tick_lo: int = 0,
    tick_hi: int | None = None,
) -> np.ndarray:
    """Return raw ADC crop shaped (n_wires, n_ticks)."""
    if tick_hi is None:
        tick_hi = waveform.shape[1]
    return waveform[wire_lo:wire_hi, tick_lo:tick_hi].astype(np.float32, copy=False)


def scale_crop(crop: np.ndarray, plane_idx: int) -> np.ndarray:
    return (crop / float(PLANE_SCALES[plane_idx])).astype(np.float32, copy=False)


def tile_patches(
    plane: np.ndarray, patch: int = PATCH
) -> tuple[np.ndarray, list[dict]]:
    """Tile a scaled plane into non-overlapping ``patch×patch`` tiles.

    Pads the *wire* axis with reflect if shorter than ``patch``.
    Crops the *tick* axis down to a multiple of ``patch`` (no pad).

    Returns
    -------
    patches : (N, patch, patch)
    meta : list of dicts with iy, ix, y0, x0 (indices into the padded/cropped plane)
    """
    h, w = plane.shape
    if h < patch:
        plane = np.pad(plane, ((0, patch - h), (0, 0)), mode="reflect")
        h = patch
    else:
        # Keep only the first floor(h/patch)*patch wires (from low-wire side of crop)
        hc = (h // patch) * patch
        plane = plane[:hc, :]
        h = hc
    wc = (w // patch) * patch
    if wc < patch:
        raise ValueError(f"Tick dim {w} smaller than patch {patch}")
    plane = plane[:, :wc]
    nh, nw = h // patch, wc // patch
    tiles = (
        plane.reshape(nh, patch, nw, patch).swapaxes(1, 2).reshape(-1, patch, patch)
    )
    meta: list[dict] = []
    for iy in range(nh):
        for ix in range(nw):
            meta.append(
                {
                    "iy": iy,
                    "ix": ix,
                    "y0": iy * patch,
                    "x0": ix * patch,
                    "patch": patch,
                }
            )
    return tiles.astype(np.float32, copy=False), meta


def write_patch_npz(
    path: Path,
    patch_arr: np.ndarray,
    *,
    record: dict,
    label: str,
    plane_idx: int,
    wire_lo: int,
    wire_hi: int,
    tick_lo: int,
    tick_hi: int,
    patch_meta: dict,
) -> None:
    """Write one 512×512 patch NPZ in the stop-signal / grid-ready schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    reco = patch_arr.astype(np.float32, copy=False)[np.newaxis, np.newaxis, ...]  # (1,1,H,W)
    np.savez_compressed(
        path,
        reco=reco,
        source_file=np.asarray(record.get("h5_path", "")),
        patch_iy=np.int32(patch_meta["iy"]),
        patch_ix=np.int32(patch_meta["ix"]),
        frame=np.int32(0),
        label=np.asarray(label),
        wire_lo=np.int32(wire_lo),
        wire_hi=np.int32(wire_hi),
        tick_lo=np.int32(tick_lo),
        tick_hi=np.int32(tick_hi),
        plane_idx=np.int32(plane_idx),
        scale=np.float32(PLANE_SCALES[plane_idx]),
        sample_id=np.asarray(record["sample_id"]),
        note=np.asarray(record.get("note", "")),
    )


def patch_filename(sample_id: str, plane_idx: int, iy: int, ix: int) -> str:
    return f"{safe_stem(sample_id)}_p{plane_idx}_y{iy}_x{ix}.npz"


def save_selected_patches(
    out_dir: Path,
    record: dict,
    label: str,
    *,
    plane_idx: int,
    wire_lo: int,
    wire_hi: int,
    tick_lo: int,
    tick_hi: int,
    patches: np.ndarray,
    patch_meta: list[dict],
    keep_mask: list[bool] | np.ndarray,
) -> list[dict]:
    """Write kept patches; return manifest rows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for keep, arr, meta in zip(keep_mask, patches, patch_meta):
        if not keep:
            continue
        name = patch_filename(record["sample_id"], plane_idx, meta["iy"], meta["ix"])
        path = out_dir / name
        write_patch_npz(
            path,
            arr,
            record=record,
            label=label,
            plane_idx=plane_idx,
            wire_lo=wire_lo,
            wire_hi=wire_hi,
            tick_lo=tick_lo,
            tick_hi=tick_hi,
            patch_meta=meta,
        )
        rows.append(
            {
                **{k: record[k] for k in record if k not in ("category",)},
                "label": label,
                "plane_idx": plane_idx,
                "wire_lo": wire_lo,
                "wire_hi": wire_hi,
                "tick_lo": tick_lo,
                "tick_hi": tick_hi,
                "patch_iy": meta["iy"],
                "patch_ix": meta["ix"],
                "npz_path": str(path.resolve()),
                "npz_name": name,
            }
        )
    return rows


def stage_to_pnfs(
    local_root: Path,
    *,
    labels: list[str] | None = None,
    pnfs_inputs: Path = PNFS_INPUTS,
    pnfs_lists: Path = PNFS_LISTS,
) -> dict[str, Path]:
    """Copy curated ``patches/{label}/*.npz`` to pnfs and write file lists.

    Returns mapping label → list path, plus ``all`` → combined list.
    """
    labels = labels or ["healthy", "unhealthy", "unique"]
    pnfs_inputs.mkdir(parents=True, exist_ok=True)
    pnfs_lists.mkdir(parents=True, exist_ok=True)
    list_paths: dict[str, Path] = {}
    all_paths: list[Path] = []

    for label in labels:
        src = local_root / "patches" / label
        if not src.is_dir():
            continue
        dest = pnfs_inputs / label
        dest.mkdir(parents=True, exist_ok=True)
        copied: list[Path] = []
        for f in sorted(src.glob("*.npz")):
            d = dest / f.name
            if not d.exists() or d.stat().st_size != f.stat().st_size:
                shutil.copy2(f, d)
            copied.append(d)
        list_path = pnfs_lists / f"handscan_curated_{label}.list"
        list_path.write_text("\n".join(str(p) for p in copied) + ("\n" if copied else ""))
        list_paths[label] = list_path
        all_paths.extend(copied)

    all_list = pnfs_lists / "handscan_curated_all.list"
    all_list.write_text("\n".join(str(p) for p in all_paths) + ("\n" if all_paths else ""))
    list_paths["all"] = all_list
    return list_paths


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
