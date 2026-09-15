#!/usr/bin/env python3
"""Live ROC accumulator for handscan validation pickles.

Watches ``inference_T100/{healthy,unhealthy}/`` and recomputes ROC / score
histograms whenever new ``*_T{T}_ddim2ddim.pkl`` files appear. Writes:

- ``handscan_validation/roc_curves/live_roc.json``
- ``handscan_validation/roc_curves/live_roc.png``
- ``handscan_validation/roc_curves/live_scores.png``
- ``handscan_validation/roc_curves/live_status.json``
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

_REPO = Path(__file__).resolve().parents[1]


def find_bundles(root: Path, T: int) -> list[Path]:
    return sorted(root.glob(f"*_T{T}_ddim2ddim.pkl"))


def patch_rms_scores(pkl_path: Path) -> list[float]:
    with pkl_path.open("rb") as fh:
        raw = pickle.load(fh)
    sal_keys = [
        k
        for p in raw.values()
        if isinstance(p, dict)
        for k in p
        if isinstance(k, str) and k.startswith("saliency-T")
    ]
    if not sal_keys:
        return []
    sal_k = sal_keys[0]
    scores = []
    for key in sorted(k for k in raw if isinstance(k, int)):
        sal = np.squeeze(raw[key][sal_k]).astype(np.float64)
        scores.append(float(np.sqrt(np.mean(np.abs(sal) ** 2))))
    return scores


def collect(root: Path, T: int, file_level: bool) -> tuple[np.ndarray, list[str]]:
    scores: list[float] = []
    names: list[str] = []
    for bp in find_bundles(root, T):
        try:
            ps = patch_rms_scores(bp)
        except Exception as e:
            print(f"[skip] {bp.name}: {e}", flush=True)
            continue
        if not ps:
            continue
        if file_level:
            scores.append(float(np.max(ps)))
            names.append(bp.name)
        else:
            scores.extend(ps)
            names.extend([f"{bp.name}#{i}" for i in range(len(ps))])
    return np.asarray(scores, dtype=float), names


def optimize_threshold(y: np.ndarray, scores: np.ndarray) -> dict:
    cuts = np.sort(np.unique(scores))[::-1]
    best = {"youden_j": -np.inf}
    for th in cuts:
        pred = scores >= th
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        tn = int(np.sum(~pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        tpr = tp / (tp + fn) if (tp + fn) else np.nan
        fpr = fp / (fp + tn) if (fp + tn) else np.nan
        pur = tp / (tp + fp) if (tp + fp) else np.nan
        j = (tpr - fpr) if np.isfinite(tpr) and np.isfinite(fpr) else -np.inf
        if j > best["youden_j"]:
            best = {
                "threshold": float(th),
                "tpr": float(tpr) if np.isfinite(tpr) else None,
                "fpr": float(fpr) if np.isfinite(fpr) else None,
                "purity": float(pur) if np.isfinite(pur) else None,
                "youden_j": float(j) if np.isfinite(j) else None,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
    return best


def update_once(
    healthy_dir: Path,
    unhealthy_dir: Path,
    out_dir: Path,
    T: int,
    file_level: bool,
) -> dict:
    h_scores, _ = collect(healthy_dir, T, file_level)
    u_scores, _ = collect(unhealthy_dir, T, file_level)

    status = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_healthy": int(len(h_scores)),
        "n_unhealthy": int(len(u_scores)),
        "n_healthy_files": len(find_bundles(healthy_dir, T)),
        "n_unhealthy_files": len(find_bundles(unhealthy_dir, T)),
        "file_level": file_level,
        "auc": None,
        "best_youden": None,
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    if len(h_scores) == 0 or len(u_scores) == 0:
        (out_dir / "live_status.json").write_text(json.dumps(status, indent=2))
        print(
            f"[live] waiting for both classes  healthy={status['n_healthy_files']} "
            f"unhealthy={status['n_unhealthy_files']}",
            flush=True,
        )
        return status

    y = np.concatenate([np.zeros(len(h_scores)), np.ones(len(u_scores))])
    scores = np.concatenate([h_scores, u_scores])
    fpr, tpr, _ = roc_curve(y, scores)
    auc_val = float(auc(fpr, tpr))
    best = optimize_threshold(y, scores)
    status["auc"] = auc_val
    status["best_youden"] = best

    payload = {
        "label": "SBND_handscan_streaks_plane1_live",
        "T": T,
        "score_mode": "file_max_rms" if file_level else "rms",
        "n_nominal_patches": int(len(h_scores)),
        "n_defect_patches": int(len(u_scores)),
        "auc": auc_val,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "best_youden": best,
        "timestamp": status["ts"],
        "n_healthy_files": status["n_healthy_files"],
        "n_unhealthy_files": status["n_unhealthy_files"],
    }
    (out_dir / "live_roc.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "live_status.json").write_text(json.dumps(status, indent=2))

    # ROC plot
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax.plot(fpr, tpr, lw=2, label=f"AUC={auc_val:.3f}")
    if best.get("fpr") is not None and best.get("tpr") is not None:
        ax.scatter([best["fpr"]], [best["tpr"]], c="crimson", zorder=5, label=f"Youden θ={best['threshold']:.4g}")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title(
        f"Live ROC  healthy={status['n_healthy_files']}  unhealthy={status['n_unhealthy_files']}"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "live_roc.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Score hist
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = 25
    ax.hist(h_scores, bins=bins, alpha=0.6, label="Healthy", color="teal")
    ax.hist(u_scores, bins=bins, alpha=0.55, label="Unhealthy", color="coral")
    if best.get("threshold") is not None:
        ax.axvline(best["threshold"], color="crimson", ls="--", label=f"θ={best['threshold']:.4g}")
    ax.set_xlabel("RMS saliency" + (" (file max)" if file_level else " (patch)"))
    ax.set_ylabel("Count")
    ax.set_title(f"Live scores  AUC={auc_val:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "live_scores.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(
        f"[live] files h={status['n_healthy_files']} u={status['n_unhealthy_files']}  "
        f"scores h={len(h_scores)} u={len(u_scores)}  AUC={auc_val:.4f}  "
        f"Youden θ={best.get('threshold')}",
        flush=True,
    )
    return status


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=Path, default=_REPO / "handscan_validation/inference_T100")
    ap.add_argument("--roc-dir", type=Path, default=_REPO / "handscan_validation/roc_curves")
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--poll-sec", type=float, default=15.0)
    ap.add_argument(
        "--file-level",
        action="store_true",
        default=True,
        help="Use max patch RMS per file (default). Better for accumulating ROC as files finish.",
    )
    ap.add_argument("--patch-level", action="store_true", help="Score every patch instead of file max")
    args = ap.parse_args()
    file_level = not args.patch_level

    healthy = args.output_root / "healthy"
    unhealthy = args.output_root / "unhealthy"
    print(f"Watching {healthy} and {unhealthy}", flush=True)

    last_key = None
    try:
        while True:
            key = (
                tuple(p.name for p in find_bundles(healthy, args.T)),
                tuple(p.name for p in find_bundles(unhealthy, args.T)),
            )
            if key != last_key:
                update_once(healthy, unhealthy, args.roc_dir, args.T, file_level)
                last_key = key
            # Also exit cleanly if orchestrator marks done and queue empty? keep watching forever unless interrupted
            time.sleep(args.poll_sec)
    except KeyboardInterrupt:
        print("Live ROC stopped.", flush=True)


if __name__ == "__main__":
    main()
