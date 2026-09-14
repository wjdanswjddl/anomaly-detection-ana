#!/usr/bin/env python3
from __future__ import annotations

import argparse
import secrets
import zlib
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve
from tqdm import tqdm


DECAY_CONST = 48.8 / 0.4

def _frame_scale(frame: np.ndarray) -> float:
    """
    Characteristic magnitude of the panel for scaling defects. Uses a floor so
    near-flat / normalized planes still get visible artifacts in defect maps.
    """
    x = frame.astype(np.float64)
    a = np.abs(x)
    rms = float(np.sqrt(np.mean(x * x)))
    return float(max(float(np.percentile(a, 99.5)), float(a.max()), rms * 3.0, 0.1))


def wave_packet(t, freq=1.0, decay=0.5, phase=0.0, amplitude=1.0, t0=0):
    t = np.asarray(t)
    return amplitude * np.exp(-decay * np.abs(t - t0)) * np.sin(
        2 * np.pi * freq * (t - t0) + phase
    )


def convolve_exp_tail(
    waveform,
    amplitude=1.0,
    decay=1.0,
    dt=1.0,
    n_tau=10,
    n_tau_extend=1.0,
    axis=0,
):
    """Causal exponential-tail convolution along `axis` (from MakeDetectorFeatures)."""
    waveform = np.asarray(waveform, dtype=float)

    n_in = waveform.shape[axis]
    n_extend = max(int(np.ceil(n_tau_extend * decay / dt)), 0)
    n_out = n_in

    if n_out > n_in:
        pad = [(0, 0)] * waveform.ndim
        pad[axis] = (0, n_out - waveform.shape[axis])
        waveform = np.pad(waveform, pad)

    t = np.arange(n_out) * dt
    kernel = np.exp(-t / decay)
    kernel *= amplitude / kernel.sum()
    kernel[0] += 1

    shape = [1] * waveform.ndim
    shape[axis] = -1
    kernel_nd = kernel.reshape(shape)

    return fftconvolve(waveform, kernel_nd, mode="full", axes=axis)


def convolve_exp_tail_only(
    waveform,
    amplitude=1.0,
    decay=1.0,
    dt=1.0,
    axis=0,
):
    """
    Causal exponential smoothing only (no identity peak on the wire).
    Kernel peak at t=0 is ``amplitude`` (not L1-normalized); avoids the tiny
    impulses that L1=amplitude normalization produced for large decay constants.
    """
    waveform = np.asarray(waveform, dtype=float)
    n_out = waveform.shape[axis]
    t = np.arange(n_out) * dt
    kernel = np.exp(-t / decay) * amplitude
    shape = [1] * waveform.ndim
    shape[axis] = -1
    kernel_nd = kernel.reshape(shape)
    return fftconvolve(waveform, kernel_nd, mode="full", axes=axis)


def apply_bad_wire(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Replace one wire (column) with random ADC-like values."""
    out = frame.copy()
    h, w = out.shape[-2], out.shape[-1]
    # reco layout matches notebook: [:, col] is one wire along drift
    col = int(rng.integers(0, w))
    maxval = _frame_scale(out) + 1e-6
    mode = rng.choice(["uniform", "normal"])
    if mode == "uniform":
        out[:, col] = rng.uniform(-1 * maxval, 1 * maxval, size=h).astype(np.float32)
    else:
        scale = rng.uniform(1.0, 3.0) * maxval
        out[:, col] = rng.normal(0.0, scale, size=h).astype(np.float32)
    return out


def apply_coh_noise(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add coherent oscillation across a vertical band (constant along wires in band)."""
    out = frame.astype(np.float32, copy=True)
    h, w = out.shape[-2], out.shape[-1]
    band = int(rng.integers(40, 80))
    c0 = int(rng.integers(0, max(1, w - band + 1)))
    sig = _frame_scale(out)
    t = np.arange(h, dtype=np.float64)
    noise = wave_packet(
        t,
        freq=rng.uniform(0.0015, 0.0035),
        decay=rng.uniform(0.006, 0.015),
        amplitude=float(rng.uniform(0.01, 0.2)) * sig,
        t0=float(rng.integers(0, h)),
        phase=rng.uniform(-np.pi, np.pi),
    ).astype(np.float32)
    band_ = noise.reshape(h, 1)
    out[:, c0 : c0 + band] = out[:, c0 : c0 + band] + band_
    return out


def apply_charge_tail(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Sparse synthetic charge deposits on a wire×time stripe, smoothed with a
    causal exponential tail along drift (axis 0). Amplitude is scaled to the
    frame RMS so it stays in-family without reusing a patch of the real image
    (which looked like texture paste when the raw patch was convolved).
    """
    out = frame.astype(np.float32, copy=True)
    h, w = out.shape[-2], out.shape[-1]
    stripe_w = int(rng.integers(35, 66))
    dst_c0 = int(rng.integers(0, max(1, w - stripe_w + 1)))
    driver = np.zeros((h, stripe_w), dtype=np.float64)
    rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))) + 1e-6
    sig = _frame_scale(frame)
    n_pulses = int(rng.integers(0, 10))
    # Keep impulses in the upper drift region so the tail fills downward in-frame.
    row_hi = max(1, min(96, h // 5))
    for _ in range(n_pulses):
        r = int(rng.integers(0, row_hi))
        c = int(rng.integers(0, stripe_w))
        q = float(rng.normal(0.0, sig * float(rng.uniform(0.35, 1.1))))
        driver[r, c] += q
    amp = float(rng.uniform(0.01, 0.2))
    tail = convolve_exp_tail_only(driver, decay=DECAY_CONST, amplitude=amp, axis=0)
    nh = tail.shape[0]
    tail512 = tail[:h].astype(np.float32) if nh >= h else np.pad(
        tail.astype(np.float32), ((0, h - nh), (0, 0))
    )
    # Match artifact strength to panel dynamics (always visible above noise floor).
    target = sig * float(rng.uniform(0.14, 0.38))
    mx = float(np.max(np.abs(tail512)) + 1e-20)
    tail512 = (tail512 * (target / mx)).astype(np.float32)
    out[:, dst_c0 : dst_c0 + stripe_w] += tail512
    return out


DEFECT_FNS = {
    "bad_wire": apply_bad_wire,
    "coh_noise": apply_coh_noise,
    "charge_tail": apply_charge_tail,
}


def sibling_output_path(npz_path: Path, defect: str) -> Path:
    parent = npz_path.parent
    out_dir = parent.parent / f"{parent.name}_{defect}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / npz_path.name


def sibling_defect_only_path(npz_path: Path, defect: str) -> Path:
    parent = npz_path.parent
    out_dir = parent.parent / f"{parent.name}_{defect}_defect"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / npz_path.name


def load_reco(npz_path: Path) -> tuple[np.ndarray, dict]:
    z = np.load(npz_path, allow_pickle=True)
    if "reco" not in z.files:
        raise KeyError(f"No 'reco' key in {npz_path}, have {z.files}")
    reco = z["reco"].astype(np.float32, copy=True)
    extra = {k: z[k] for k in z.files if k != "reco"}
    return reco, extra


def save_npz(out_path: Path, reco: np.ndarray, extra: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    to_save = {"reco": reco, **extra}
    np.savez_compressed(out_path, **to_save)


def save_defect_only_npz(out_path: Path, defect_delta: np.ndarray, extra: dict) -> None:
    """Defect contribution only: modified_reco - original_reco (key 'defect')."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    to_save = {"defect": defect_delta.astype(np.float32, copy=False), **extra}
    np.savez_compressed(out_path, **to_save)


def process_file(npz_path: Path, defect: str, path_seed: int) -> Path:
    reco, extra = load_reco(npz_path)
    fn = DEFECT_FNS[defect]
    # reco: (n_frames, 1, H, W) or (n_frames, H, W)
    out = np.empty_like(reco)
    dcode = {"bad_wire": 1, "coh_noise": 2, "charge_tail": 3}.get(defect, 0)
    if reco.ndim == 4 and reco.shape[1] == 1:
        for i in range(reco.shape[0]):
            sub = np.random.default_rng(path_seed + dcode * 1_000_003 + i * 17)
            out[i, 0] = fn(reco[i, 0], sub)
    elif reco.ndim == 3:
        for i in range(reco.shape[0]):
            sub = np.random.default_rng(path_seed + dcode * 1_000_003 + i * 17)
            out[i] = fn(reco[i], sub)
    elif reco.ndim == 2:
        out = fn(reco, np.random.default_rng(path_seed + dcode * 1_000_003))
    else:
        raise ValueError(f"Unexpected reco shape {reco.shape} in {npz_path}")
    delta = out.astype(np.float32, copy=False) - reco.astype(np.float32, copy=False)
    outp = sibling_output_path(npz_path, defect)
    save_npz(outp, out, extra)
    save_defect_only_npz(sibling_defect_only_path(npz_path, defect), delta, extra)
    return outp


_SKIP_FILENAME_SUBSTRINGS = ("tpc2-plane1", "tpc1-plane1")


def _skip_npz(npz_path: Path) -> bool:
    name = npz_path.name
    return any(s in name for s in _SKIP_FILENAME_SUBSTRINGS)


def _path_seed_component(npz_path: Path) -> int:
    """Stable per-path integer (unlike built-in hash(), which varies per process)."""
    return zlib.adler32(str(npz_path.resolve()).encode()) & 0x7FFFFFFF


def iter_npz_roots(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for r in roots:
        if r.is_file() and r.suffix == ".npz":
            files.append(r)
        elif r.is_dir():
            files.extend(sorted(r.rglob("*.npz")))
    return files


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="npz files and/or directories to scan for *.npz",
    )
    p.add_argument(
        "--defects",
        nargs="+",
        default=list(DEFECT_FNS.keys()),
        choices=list(DEFECT_FNS.keys()),
        help="Which defects to apply (each writes its own output tree)",
    )
    p.add_argument("--seed", type=int, default=None, help="Base RNG seed")
    args = p.parse_args()

    all_npz = iter_npz_roots(args.inputs)
    npz_files = [p for p in all_npz if not _skip_npz(p)]
    if not npz_files:
        if all_npz:
            raise SystemExit(
                "All .npz files matched skip patterns "
                f"({_SKIP_FILENAME_SUBSTRINGS})."
            )
        raise SystemExit("No .npz files found.")

    base_seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    for defect in args.defects:
        for npz_path in tqdm(npz_files, desc=defect):
            path_seed = (base_seed + _path_seed_component(npz_path)) & (2**32 - 1)
            process_file(npz_path, defect, path_seed)


if __name__ == "__main__":
    main()
