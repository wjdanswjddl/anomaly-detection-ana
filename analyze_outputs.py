"""Batch analysis of diffusion inference outputs.

Iterates over all combinations of DEFECT_TYPE, MODEL_TYPE, and MODE,
computing patch-level scores, ROC curves, and saving figures + JSON results.

Usage
-----
Run with defaults (all combinations):
    python analyze_outputs.py

Restrict to specific values:
    python analyze_outputs.py \
        --defect-types bad_wire \
        --model-types "" -anisotropic \
        --modes rand2ddim ddim2ddim

Override output directories:
    python analyze_outputs.py --roc-save-dir roc_curves --fig-save-dir figures_paper
"""
from __future__ import annotations

import argparse
import datetime
import json
import pickle
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for batch use
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import auc, roc_curve
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Defaults — edit or override via CLI flags
# ---------------------------------------------------------------------------
DEFAULT_DEFECT_TYPES = ["bad_wire", "coh_noise", "charge_tail"]
DEFAULT_MODEL_TYPES  = ["", "-ramp","-anisotropic", "-cosine"]
DEFAULT_MODES        = ["ddim2ddim", "rand2ddim", "rand2ddpm"] #, "ddpm2ddpm", "ddim2ddpm"]
DEFAULT_T            = 200
DEFAULT_SCORE        = "rms"   # "rms" | "proj_max_max"
DEFAULT_PROJECTION_AXIS = 0
DEFAULT_ROC_SAVE_DIR = Path("roc_curves")
DEFAULT_FIG_SAVE_DIR = Path("figures_paper")
STYLEFILE            = "presentation.mplstyle"
BASE_DATA_DIR        = Path("/scratch/7DayLifetime/munjung/ICARUS")

# ---------------------------------------------------------------------------
# Helpers (ported verbatim from notebook)
# ---------------------------------------------------------------------------

def find_bundles(root: Path, T: int, mode: str = "ddim2ddim") -> list[Path]:
    """Find all pickle bundles for a given mode and timestep T."""
    return sorted(root.glob(f"*_T{T}_{mode}.pkl"))


def split_bundle(raw: dict) -> tuple[dict[int, dict], dict]:
    trace_keys = [k for k in raw if isinstance(k, str) and k.startswith("__") and k.endswith("_source__")]
    meta = raw.get(trace_keys[0], {}) if trace_keys else {}
    per_patch = {k: v for k, v in raw.items() if isinstance(k, int)}
    return per_patch, meta


def infer_keys(sample_patch: dict) -> tuple[str, str]:
    reco = [
        k for k in sample_patch
        if isinstance(k, str) and "-T" in k and not k.startswith("saliency") and k != "original"
    ]
    sal = [k for k in sample_patch if isinstance(k, str) and k.startswith("saliency-T")]
    if len(reco) != 1 or len(sal) != 1:
        raise ValueError(
            f"Could not uniquely infer reconstruction keys from first patch "
            f"(found reco={reco}, saliency={sal})"
        )
    return reco[0], sal[0]


def load_bundle(path: Path) -> tuple[dict[int, dict], dict, str, str]:
    with path.open("rb") as fh:
        raw = pickle.load(fh)
    per_patch, trace = split_bundle(raw)
    if not per_patch:
        raise ValueError(f"No patch entries in {path}")
    ddim_k, sal_k = infer_keys(per_patch[next(iter(per_patch))])
    return per_patch, trace, ddim_k, sal_k


def projected_profile(abs_sal: np.ndarray, axis: int) -> np.ndarray:
    return np.mean(abs_sal, axis=axis)


def bundle_patch_scores(bundle_path: Path, axis: int, score_mode: str) -> list[float]:
    """Return one score per patch inside this bundle."""
    per_patch, _, _, sal_k = load_bundle(bundle_path)
    patch_scores: list[float] = []
    for j in sorted(per_patch.keys()):
        sal = np.squeeze(per_patch[j][sal_k]).astype(np.float64)
        abs_s = np.abs(sal)
        if score_mode == "rms":
            patch_scores.append(float(np.sqrt(np.mean(abs_s**2))))
        elif score_mode == "proj_max_max":
            proj = projected_profile(abs_s, axis)
            patch_scores.append(float(np.max(proj)))
        else:
            raise ValueError(score_mode)
    return patch_scores


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_dirs(
    defect_type: str,
    model_type: str,
    mode: str,
    base: Path = BASE_DATA_DIR,
) -> tuple[Path, Path]:
    """Return (nominal_dir, defect_dir) for a given combination."""
    if mode == "ddim2ddim":
        nominal = base / f"plane1_healthy_outputs{model_type}"
        defects = base / f"plane1_{defect_type}_outputs{model_type}"
    else:
        nominal = base / f"plane1_healthy_outputs-{mode}{model_type}" / f"mixed_{mode}"
        defects = base / f"plane1_{defect_type}_outputs-{mode}{model_type}" / f"mixed_{mode}"
    return nominal, defects


# ---------------------------------------------------------------------------
# Per-combination analysis
# ---------------------------------------------------------------------------

def run_combination(
    *,
    defect_type: str,
    model_type: str,
    mode: str,
    T: int,
    score: str,
    projection_axis: int,
    roc_save_dir: Path | None,
    fig_save_dir: Path | None,
) -> None:
    nominal_dir, defects_dir = resolve_dirs(defect_type, model_type, mode)

    tag = f"{defect_type}{model_type}_{mode}_T{T}"
    print(f"\n{'='*70}")
    print(f"  DEFECT_TYPE={defect_type!r}  MODEL_TYPE={model_type!r}  MODE={mode!r}  T={T}")
    print(f"  nominal : {nominal_dir}")
    print(f"  defects : {defects_dir}")

    for label, d in [("nominal", nominal_dir), ("defects", defects_dir)]:
        if not d.exists():
            print(f"  [SKIP] {label} dir not found: {d}")
            return
        n_files = sum(1 for _ in d.iterdir())
        if n_files < 5:
            print(f"  [SKIP] {label} dir has only {n_files} file(s) (< 5): {d}")
            return

    nom_pkls = find_bundles(nominal_dir, T, mode)
    def_pkls = find_bundles(defects_dir, T, mode)
    print(f"  Bundles: nominal={len(nom_pkls)}  defect={len(def_pkls)}")

    if not nom_pkls or not def_pkls:
        print("  [SKIP] need at least one bundle per class")
        return

    # Score all patches
    nom_scores: list[float] = []
    def_scores: list[float] = []

    for bp in tqdm(nom_pkls, desc=f"  score nominal [{tag}]", leave=False):
        try:
            nom_scores.extend(bundle_patch_scores(bp, projection_axis, score))
        except Exception as exc:
            print(f"  skip nominal {bp.name}: {exc}")

    for bp in tqdm(def_pkls, desc=f"  score defect  [{tag}]", leave=False):
        try:
            def_scores.extend(bundle_patch_scores(bp, projection_axis, score))
        except Exception as exc:
            print(f"  skip defect  {bp.name}: {exc}")

    nom_arr = np.asarray(nom_scores, dtype=float)
    def_arr = np.asarray(def_scores, dtype=float)
    y = np.concatenate([np.zeros(len(nom_arr)), np.ones(len(def_arr))])
    scores_all = np.concatenate([nom_arr, def_arr])

    print(f"  n nominal patches={len(nom_arr)}  n defect patches={len(def_arr)}")

    if not (scores_all.size and np.unique(y).size >= 2):
        print("  [SKIP] need nonempty scores for both classes")
        return

    # ROC
    fpr, tpr, _ = roc_curve(y, scores_all)
    auc_val = auc(fpr, tpr)
    print(f"  AUC = {auc_val:.4f}")

    # Efficiency × purity
    thresholds = np.sort(
        np.unique(np.concatenate(([scores_all.min() - 1e-6], scores_all, [scores_all.max() + 1e-6])))
    )
    eff, pur = [], []
    for th in thresholds[::-1]:
        pred = scores_all >= th
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        tp_fn = tp + np.sum(~pred & (y == 1))
        eff.append(tp / tp_fn if tp_fn else np.nan)
        pur.append(tp / (tp + fp) if (tp + fp) else np.nan)
    eff = np.asarray(eff)
    pur = np.asarray(pur)

    # ── Save ROC JSON ────────────────────────────────────────────────────────
    if roc_save_dir is not None:
        roc_save_dir.mkdir(parents=True, exist_ok=True)
        roc_label = f"ICARUS_plane1_{defect_type}{model_type}"
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        out_path = roc_save_dir / f"{roc_label}_T{T}_{score}_{mode}.json"
        payload = {
            "label": roc_label,
            "defect_type": defect_type,
            "model_type": model_type,
            "mode": mode,
            "T": T,
            "score_mode": score,
            "projection_axis": projection_axis,
            "nominal_dir": str(nominal_dir),
            "defect_dir": str(defects_dir),
            "n_nominal_patches": int(len(nom_arr)),
            "n_defect_patches": int(len(def_arr)),
            "auc": float(auc_val),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "timestamp": timestamp,
        }
        with out_path.open("w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"  Saved ROC JSON → {out_path}")

    # ── Figures ──────────────────────────────────────────────────────────────
    if fig_save_dir is not None:
        fig_save_dir.mkdir(parents=True, exist_ok=True)

        # ROC + efficiency×purity
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
        ax[0].plot(fpr, tpr, lw=2, label=f"ROC AUC={auc_val:.4f}")
        ax[0].plot([0, 1], [0, 1], linestyle="--", color="grey")
        ax[0].set_xlabel("False-positive rate")
        ax[0].set_ylabel("True-positive rate (efficiency vs nominal)")
        ax[0].set_title(f"ROC — {defect_type}{model_type} {mode}")
        ax[0].legend()
        ax[0].set_aspect("equal", adjustable="box")

        sel = ~(np.isnan(eff) | np.isnan(pur))
        ax[1].plot(eff[sel], pur[sel], lw=2)
        ax[1].set_xlim(0.0, 1.03)
        ax[1].set_ylim(0.0, 1.03)
        ax[1].set_xlabel("Efficiency (fraction of defects flagged)")
        ax[1].set_ylabel("Purity TP/(TP+FP) among flagged samples")
        ax[1].set_title("Efficiency × purity")
        ax[1].grid(True, alpha=0.3)
        plt.tight_layout()
        roc_fig_path = fig_save_dir / f"roc-{tag}.pdf"
        plt.savefig(roc_fig_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"  Saved ROC figure → {roc_fig_path}")

        # Score overlap histogram
        range_max = max(np.max(nom_arr), np.max(def_arr))
        range_min = min(np.min(nom_arr), np.min(def_arr))
        bins = np.linspace(range_min, range_max, 26)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(
            nom_arr, bins=bins, alpha=0.6, label="Healthy", density=False,
            weights=np.ones_like(nom_arr) * 5, color="teal",
        )
        ax.hist(
            def_arr, bins=bins, alpha=0.55, label=f"Defect ({defect_type})", density=False,
            weights=np.ones_like(def_arr) * 5, color="coral",
        )
        ax.set_xlabel("Score")
        ax.set_ylabel("Count (Area Normalized)")
        ax.legend()
        ax.set_title(f"{mode}  |  {defect_type}{model_type}")
        plt.tight_layout()
        overlap_fig_path = fig_save_dir / f"score_overlap-{tag}.pdf"
        plt.savefig(overlap_fig_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"  Saved overlap figure → {overlap_fig_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch ROC analysis over DEFECT_TYPE × MODEL_TYPE × MODE combinations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--defect-types", nargs="+", default=DEFAULT_DEFECT_TYPES,
        metavar="TYPE",
        help="Defect type labels (e.g. bad_wire stuck_wire).",
    )
    parser.add_argument(
        "--model-types", nargs="+", default=DEFAULT_MODEL_TYPES,
        metavar="MTYPE",
        help='Model type suffixes (e.g. "" or "-anisotropic"). Use empty string for baseline.',
    )
    parser.add_argument(
        "--modes", nargs="+", default=DEFAULT_MODES,
        metavar="MODE",
        help="Diffusion modes to evaluate.",
    )
    parser.add_argument("--T", type=int, default=DEFAULT_T, help="Timestep T used in filenames.")
    parser.add_argument(
        "--score", default=DEFAULT_SCORE, choices=["rms", "proj_max_max"],
        help="Patch-level scoring method.",
    )
    parser.add_argument(
        "--projection-axis", type=int, default=DEFAULT_PROJECTION_AXIS,
        help="Axis along which to project |saliency| (0 = per-row, 1 = per-column).",
    )
    parser.add_argument(
        "--roc-save-dir", type=Path, default=DEFAULT_ROC_SAVE_DIR,
        help="Directory for ROC JSON files. Pass 'none' to disable.",
    )
    parser.add_argument(
        "--fig-save-dir", type=Path, default=DEFAULT_FIG_SAVE_DIR,
        help="Directory for saved figures. Pass 'none' to disable.",
    )
    parser.add_argument(
        "--no-style", action="store_true",
        help="Skip loading presentation.mplstyle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.no_style and Path(STYLEFILE).exists():
        plt.style.use(STYLEFILE)

    roc_save_dir = None if str(args.roc_save_dir).lower() == "none" else args.roc_save_dir
    fig_save_dir = None if str(args.fig_save_dir).lower() == "none" else args.fig_save_dir

    combos = list(product(args.defect_types, args.model_types, args.modes))
    print(f"Running {len(combos)} combination(s): "
          f"{len(args.defect_types)} defect × {len(args.model_types)} model × {len(args.modes)} mode")

    for defect_type, model_type, mode in combos:
        run_combination(
            defect_type=defect_type,
            model_type=model_type,
            mode=mode,
            T=args.T,
            score=args.score,
            projection_axis=args.projection_axis,
            roc_save_dir=roc_save_dir,
            fig_save_dir=fig_save_dir,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
