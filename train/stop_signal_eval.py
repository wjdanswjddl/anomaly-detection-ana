"""Find-stop-signal helpers from **grid job outputs** (CPU only).

Used by ``train/03_LearningCurves.ipynb``. Never imports torch / never runs
inference — only unpacks ``out_*.tgz`` and scores ``*_arrays.npz``.
"""
from __future__ import annotations

import json
import re
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# Coh (default):   <stamp>__stopsignal_{config}_ckpt{step}_T{tag}
# Healthy:         <stamp>__stopsignal_healthy_{config}_ckpt{step}_T{tag}
# Legacy probes:   <stamp>__stopsignal_{coh|both}_ckpt{step}_T{tag}
_CAMPAIGN_RE = re.compile(
    r"^(?P<stamp>\d{4}_\d{2}_\d{2}_\d{6})__stopsignal_"
    r"(?:(?P<probe>healthy|both)_)?"
    r"(?P<config>linear|anisotropic|gray|cosine|ramp|pred_xstart|coh)_"
    r"ckpt(?P<step>\d+)_T(?P<t_tag>.+)$"
)


@dataclass(frozen=True)
class CampaignRef:
    path: Path
    config: str  # linear | anisotropic | ...
    step: int
    t_tag: str  # e.g. "200" or "100-200"
    stamp: str
    n_tgz: int
    probe: str = "coh"  # coh | healthy | both

    @property
    def timesteps(self) -> list[int]:
        return [int(x) for x in self.t_tag.split("-") if x.isdigit()]


# ---------------------------------------------------------------------------
# Campaign discovery / unpack
# ---------------------------------------------------------------------------


def discover_stopsignal_campaigns(
    pnfs_inference: Path,
    *,
    configs: tuple[str, ...] = ("linear", "anisotropic", "gray"),
    probes: tuple[str, ...] = ("coh", "healthy"),
    require_tgz: bool = True,
) -> list[CampaignRef]:
    """List stop-signal campaign dirs under pnfs ``out/inference``."""
    if not pnfs_inference.is_dir():
        return []
    out: list[CampaignRef] = []
    for d in sorted(pnfs_inference.iterdir()):
        if not d.is_dir():
            continue
        m = _CAMPAIGN_RE.match(d.name)
        if m is None:
            continue
        cfg = m.group("config")
        probe = m.group("probe") or ("coh" if cfg != "coh" else "coh")
        if cfg == "coh":
            # legacy naming: stopsignal_coh_ckpt* — skip unless explicitly wanted
            continue
        if cfg not in configs:
            continue
        if probe not in probes:
            continue
        n_tgz = len(list(d.glob("out_*.tgz")))
        if require_tgz and n_tgz == 0:
            continue
        out.append(
            CampaignRef(
                path=d,
                config=cfg,
                step=int(m.group("step")),
                t_tag=m.group("t_tag"),
                stamp=m.group("stamp"),
                n_tgz=n_tgz,
                probe=probe,
            )
        )
    return out


def prefer_campaigns(
    campaigns: list[CampaignRef],
    *,
    prefer_t_tag: str | None = "100-200",
    min_tgz: int = 1,
) -> dict[tuple[str, str, int], CampaignRef]:
    """One campaign per (probe, config, step). Prefer ``prefer_t_tag`` when complete.

    A preferred-tag campaign only replaces an existing entry when
    ``n_tgz >= min_tgz`` (use 12 for a full patch probe). Incomplete
    multi-T jobs therefore do not displace finished single-T campaigns.
    """
    best: dict[tuple[str, str, int], CampaignRef] = {}
    for c in campaigns:
        key = (c.probe, c.config, c.step)
        prev = best.get(key)
        if prev is None:
            best[key] = c
            continue
        prefer_ok = (
            prefer_t_tag
            and c.t_tag == prefer_t_tag
            and prev.t_tag != prefer_t_tag
            and c.n_tgz >= min_tgz
        )
        if prefer_ok:
            best[key] = c
            continue
        if prefer_t_tag and prev.t_tag == prefer_t_tag and c.t_tag != prefer_t_tag:
            if prev.n_tgz < min_tgz and c.n_tgz > prev.n_tgz:
                best[key] = c
            continue
        if c.n_tgz > prev.n_tgz:
            best[key] = c
        elif c.n_tgz == prev.n_tgz and c.stamp > prev.stamp:
            best[key] = c
    return best


def unpack_campaign(
    campaign: Path,
    unpack_root: Path,
    *,
    force: bool = False,
) -> Path:
    """Extract all ``out_*.tgz`` into ``unpack_root / campaign.name``."""
    dest = unpack_root / campaign.name
    dest.mkdir(parents=True, exist_ok=True)
    for tgz in sorted(campaign.glob("out_*.tgz")):
        job = tgz.stem  # out_N
        job_dir = dest / job
        marker = job_dir / ".unpacked"
        if marker.is_file() and not force:
            continue
        job_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tgz, "r:gz") as tar:
            # Python 3.12+ may require filter=; ignore if unsupported.
            try:
                tar.extractall(path=dest, filter="data")
            except TypeError:
                tar.extractall(path=dest)
        marker.write_text("ok\n")
    return dest


# ---------------------------------------------------------------------------
# Scoring (CPU)
# ---------------------------------------------------------------------------


def noise_region_mask(original: np.ndarray, abs_quantile: float = 0.70) -> np.ndarray:
    x = np.asarray(original, dtype=np.float64)
    thr = float(np.quantile(np.abs(x), abs_quantile))
    return np.abs(x) <= thr


def patch_rms(saliency: np.ndarray) -> float:
    s = np.asarray(saliency, dtype=np.float64)
    return float(np.sqrt(np.mean(s * s)))


def patch_noise_region_rms(
    saliency: np.ndarray, original: np.ndarray, abs_quantile: float = 0.70
) -> float:
    m = noise_region_mask(original, abs_quantile=abs_quantile)
    s = np.asarray(saliency, dtype=np.float64)
    if int(m.sum()) < 16:
        return patch_rms(s)
    return float(np.sqrt(np.mean((s[m]) ** 2)))


def spectrum_1d(plane: np.ndarray, axis: int) -> np.ndarray:
    x = np.asarray(plane, dtype=np.float64)
    return np.abs(np.fft.rfft(x, axis=axis)).sum(axis=1 - axis)[1:]


def fft_l2_distance(
    original: np.ndarray,
    reconstructed: np.ndarray,
    axis: int,
    midband: tuple[int, int] | None = None,
) -> float:
    a = spectrum_1d(original, axis)
    b = spectrum_1d(reconstructed, axis)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if midband is not None:
        lo, hi = midband
        a, b = a[max(0, lo) : min(n, hi)], b[max(0, lo) : min(n, hi)]
    if len(a) == 0:
        return float("nan")
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))


def secondary_proxy_dict(
    original: np.ndarray,
    reconstructed: np.ndarray,
    *,
    abs_quantile: float = 0.70,
    wire_midband: tuple[int, int] = (50, 150),
) -> dict[str, float]:
    orig = np.squeeze(original).astype(np.float64)
    reco = np.squeeze(reconstructed).astype(np.float64)
    sal = reco - orig
    m = noise_region_mask(orig, abs_quantile=abs_quantile)
    return {
        "noise_resid_rms": float(np.sqrt(np.mean((sal[m]) ** 2))) if m.any() else float("nan"),
        "full_resid_rms": patch_rms(sal),
        "noise_frac": float(m.mean()),
        "fft_time_rel_l2": fft_l2_distance(orig, reco, axis=0),
        "fft_wire_rel_l2": fft_l2_distance(orig, reco, axis=1),
        "fft_wire_midband_rel_l2": fft_l2_distance(orig, reco, axis=1, midband=wire_midband),
        "bg_texture_std_orig": float(np.std(orig[m])) if m.any() else float("nan"),
        "bg_texture_std_reco": float(np.std(reco[m])) if m.any() else float("nan"),
        "bg_texture_std_ratio": (
            float(np.std(reco[m]) / (np.std(orig[m]) + 1e-12)) if m.any() else float("nan")
        ),
    }


def roc_auc_youden(
    nominal_scores: np.ndarray, anomaly_scores: np.ndarray
) -> dict[str, float | None]:
    """Binary ROC: nominal=0, anomaly=1, higher score → more anomalous. Pure NumPy."""
    if len(nominal_scores) == 0 or len(anomaly_scores) == 0:
        return {
            "auc": None,
            "youden_j": None,
            "threshold": None,
            "tpr": None,
            "fpr": None,
        }
    y = np.concatenate(
        [np.zeros(len(nominal_scores), dtype=np.int8), np.ones(len(anomaly_scores), dtype=np.int8)]
    )
    s = np.concatenate(
        [np.asarray(nominal_scores, dtype=float), np.asarray(anomaly_scores, dtype=float)]
    )
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    n_pos = float(tps[-1])
    n_neg = float(fps[-1])
    if n_pos <= 0 or n_neg <= 0:
        return {
            "auc": None,
            "youden_j": None,
            "threshold": None,
            "tpr": None,
            "fpr": None,
        }
    tpr = tps / n_pos
    fpr = fps / n_neg
    # Pad endpoints for trapezoidal AUC
    tpr_pad = np.concatenate([[0.0], tpr, [1.0]])
    fpr_pad = np.concatenate([[0.0], fpr, [1.0]])
    trapz = getattr(np, "trapezoid", None) or np.trapz
    auc_val = float(trapz(tpr_pad, fpr_pad))
    j = tpr - fpr
    i = int(np.nanargmax(j))
    return {
        "auc": float(auc_val),
        "youden_j": float(j[i]),
        "threshold": float(s_sorted[i]),
        "tpr": float(tpr[i]),
        "fpr": float(fpr[i]),
    }


def early_stop_verdict(
    steps: list[int],
    metric: list[float | None],
    *,
    patience_steps: int = 20000,
    min_delta: float = 0.005,
    metric_name: str = "AUC",
) -> dict:
    pairs = [(s, m) for s, m in zip(steps, metric) if m is not None and np.isfinite(m)]
    if len(pairs) < 2:
        return {
            "stop": False,
            "reason": f"Need ≥2 finite {metric_name} points.",
            "best_step": pairs[0][0] if pairs else None,
            "best_value": pairs[0][1] if pairs else None,
        }
    best_step, best_val = max(pairs, key=lambda p: p[1])
    last_step, last_val = pairs[-1]
    stalled = (last_step - best_step) >= patience_steps and (best_val - last_val) < min_delta
    if (last_val - best_val) >= min_delta and last_step > best_step:
        best_step, best_val = last_step, last_val
        stalled = False
    return {
        "stop": bool(stalled),
        "reason": (
            f"No {metric_name} gain ≥{min_delta} for ≥{patience_steps} steps after best "
            f"@{best_step} ({metric_name}={best_val:.6g})."
            if stalled
            else f"Keep training; best so far step={best_step} {metric_name}={best_val:.6g} "
            f"(last step={last_step} {metric_name}={last_val:.6g})."
        ),
        "best_step": int(best_step),
        "best_value": float(best_val),
        "last_step": int(last_step),
        "last_value": float(last_val),
    }


# ---------------------------------------------------------------------------
# Label helpers + campaign scoring
# ---------------------------------------------------------------------------


def classify_patch_label(
    filename: str,
    *,
    healthy_names: set[str] | None = None,
    anomaly_names: set[str] | None = None,
) -> str:
    """Return ``healthy`` | ``anomaly`` | ``unknown`` from filename / optional sets."""
    stem = Path(filename).name
    if healthy_names and stem in healthy_names:
        return "healthy"
    if anomaly_names and stem in anomaly_names:
        return "anomaly"
    low = stem.lower()
    if "coh_noise" in low or "bad_wire" in low or "charge_tail" in low:
        return "anomaly"
    if "healthy" in low:
        return "healthy"
    # Patch-level stop-signal probe: anomaly campaigns only contain defect files
    # with the same names as healthy twins — label must come from name sets.
    return "unknown"


def load_name_set(directory: Path | None) -> set[str]:
    if directory is None or not directory.is_dir():
        return set()
    return {p.name for p in directory.glob("*.npz")}


def iter_arrays_npz(unpacked_campaign: Path, T: int, mode: str = "ddim2ddim") -> list[Path]:
    return sorted(unpacked_campaign.rglob(f"*_T{T}_{mode}_arrays.npz"))


def score_arrays_file(
    npz_path: Path,
    *,
    score_kind: str = "noise_rms",
    abs_quantile: float = 0.70,
) -> tuple[list[float], list[dict[str, float]], str]:
    """Return (patch_scores, secondary_proxies_per_patch, input_filename)."""
    z = np.load(npz_path)
    orig = z["original"]
    reco = z["reconstructed"]
    sal = z["saliency"] if "saliency" in z.files else reco - orig
    fname = str(z["input_filename"]) if "input_filename" in z.files else npz_path.name
    scores: list[float] = []
    prox: list[dict[str, float]] = []
    n = orig.shape[0]
    for i in range(n):
        o = np.squeeze(orig[i])
        r = np.squeeze(reco[i])
        s = np.squeeze(sal[i])
        if score_kind == "rms":
            scores.append(patch_rms(s))
        else:
            scores.append(patch_noise_region_rms(s, o, abs_quantile=abs_quantile))
        prox.append(secondary_proxy_dict(o, r, abs_quantile=abs_quantile))
    return scores, prox, Path(fname).name


def evaluate_unpacked_campaign(
    unpacked: Path,
    *,
    T: int,
    score_kind: str = "noise_rms",
    healthy_names: set[str] | None = None,
    anomaly_names: set[str] | None = None,
    default_label: str = "anomaly",
) -> dict:
    """Score all arrays.npz at timestep ``T``; compute AUC if both classes present."""
    files = iter_arrays_npz(unpacked, T)
    healthy_scores: list[float] = []
    anomaly_scores: list[float] = []
    unknown_scores: list[float] = []
    secondary_rows: list[dict] = []

    for fp in files:
        scores, prox_list, fname = score_arrays_file(fp, score_kind=score_kind)
        label = classify_patch_label(
            fname, healthy_names=healthy_names, anomaly_names=anomaly_names
        )
        if label == "unknown":
            label = default_label
        for sc, prox in zip(scores, prox_list):
            if label == "healthy":
                healthy_scores.append(sc)
            elif label == "anomaly":
                anomaly_scores.append(sc)
            else:
                unknown_scores.append(sc)
            secondary_rows.append({"file": fname, "label": label, **prox})

    metrics = roc_auc_youden(
        np.asarray(healthy_scores, dtype=float),
        np.asarray(anomaly_scores, dtype=float),
    )
    return {
        "T": T,
        "score_kind": score_kind,
        "n_arrays": len(files),
        "n_healthy": len(healthy_scores),
        "n_anomaly": len(anomaly_scores),
        "n_unknown": len(unknown_scores),
        "healthy_mean": float(np.mean(healthy_scores)) if healthy_scores else None,
        "anomaly_mean": float(np.mean(anomaly_scores)) if anomaly_scores else None,
        "anomaly_std": float(np.std(anomaly_scores)) if anomaly_scores else None,
        **metrics,
        "secondary_mean": _mean_prox(secondary_rows),
    }


def _mean_prox(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    keys = [k for k in rows[0] if k not in ("file", "label")]
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        raise TypeError(type(obj))

    path.write_text(json.dumps(payload, indent=2, default=_default))


def load_json(path: Path):
    return json.loads(path.read_text())
