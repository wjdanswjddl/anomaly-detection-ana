"""Unified paths and naming for the anomaly-detection workflow.

Code lives under APP_ROOT (this repo). Large products and outputs live under
DATA_ROOT (parallel /exp/sbnd/data tree). GPU training / bulk inference on EAF
uses SCRATCH_ROOT when present.

Import from notebooks/scripts::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(".../anomaly-detection")))  # repo root
    from configs.paths import APP_ROOT, DATA_ROOT, ensure_layout, resolve_inference_dirs
"""
from __future__ import annotations

import os
import socket
from pathlib import Path

# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------

APP_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/exp/sbnd/data/users/munjung/anomaly-detection")

# EAF / GPU node scratch (7-day lifetime). Absent on sbndbuild*.
SCRATCH_ROOT = Path(
    os.environ.get(
        "ANOMALY_SCRATCH_ROOT",
        "/scratch/7DayLifetime/munjung/anomaly-detection",
    )
)

# Shared checkpoints from gputnam (read-only on /exp)
GPUTNAM_TRAINING = Path("/exp/sbnd/data/users/gputnam/training-SBND")

# Vendored / cloned diffusion training code (prefer EAF clone when discovered)
DIFFUSION_ANOMALY_APP = APP_ROOT / "train" / "diffusion-anomaly"
DIFFUSION_ANOMALY_SIBLING = Path("/exp/sbnd/app/users/munjung/diffusion-anomaly")

# ---------------------------------------------------------------------------
# Data-area layout (outputs)
# ---------------------------------------------------------------------------

SAMPLES_H5 = DATA_ROOT / "samples" / "h5"
SAMPLES_NPZ = DATA_ROOT / "samples" / "npz"
SAMPLES_HANDSCAN = DATA_ROOT / "samples" / "handscan"
SAMPLES_DEFECTS = DATA_ROOT / "samples" / "defects"
HANDSCAN_VALIDATION = SAMPLES_HANDSCAN / "validation"

INFERENCE_OUT = DATA_ROOT / "inference"
TRAINING_OUT = DATA_ROOT / "training"
FIGURES_OUT = DATA_ROOT / "figures"
ROC_OUT = DATA_ROOT / "roc_curves"
ARCHIVE_DATA = DATA_ROOT / "archive"

# Scratch layout (EAF)
SCRATCH_NPZ = SCRATCH_ROOT / "npz"
SCRATCH_INFERENCE = SCRATCH_ROOT / "inference"
SCRATCH_TRAINING = SCRATCH_ROOT / "training"
SCRATCH_ICARUS = Path("/scratch/7DayLifetime/munjung/ICARUS")  # legacy ICARUS runs

# ---------------------------------------------------------------------------
# Naming conventions
# ---------------------------------------------------------------------------

# Inference pickle: ``{stem}_T{T}_{mode}.pkl``
INFERENCE_MODES = (
    "ddim2ddim",
    "rand2ddim",
    "rand2ddpm",
    "ddpm2ddpm",
    "ddim2ddpm",
)

# Synthetic defect tags (apply_detector_defects_npz / MakeDetectorFeatures)
DEFECT_TYPES = ("bad_wire", "coh_noise", "charge_tail")

# Model / schedule suffix used in output directory names
# (matches configs.train_configs.TRAIN_CONFIGS model_type_suffix)
MODEL_TYPES = ("", "-ramp", "-anisotropic", "-cosine", "-pred_xstart")

# Canonical training config names (see configs/train_flags/)
TRAIN_CONFIG_NAMES = ("linear", "cosine", "ramp", "anisotropic", "pred_xstart")

DEFAULT_T = 200
DEFAULT_SCORE = "rms"  # rms | proj_max_max


def hostname() -> str:
    return socket.gethostname()


def on_eaf() -> bool:
    """Heuristic: EAF nodes expose /scratch/7DayLifetime and conda diffusion env."""
    return SCRATCH_ROOT.parent.exists() or Path("/home/munjung/.conda/envs/diffusion").exists()


def diffusion_code_root() -> Path:
    """Best available guided-diffusion / diffusion-anomaly checkout."""
    for candidate in (
        Path(os.environ["DIFFUSION_ANOMALY_ROOT"])
        if os.environ.get("DIFFUSION_ANOMALY_ROOT")
        else None,
        DIFFUSION_ANOMALY_APP,
        DIFFUSION_ANOMALY_SIBLING,
    ):
        if candidate is not None and (candidate / "guided_diffusion").is_dir():
            return candidate
    return DIFFUSION_ANOMALY_APP


def ensure_layout() -> None:
    """Create expected output directories under DATA_ROOT (and scratch if present)."""
    for p in (
        SAMPLES_H5,
        SAMPLES_NPZ,
        SAMPLES_HANDSCAN,
        SAMPLES_DEFECTS,
        HANDSCAN_VALIDATION,
        INFERENCE_OUT,
        TRAINING_OUT / "checkpoints",
        TRAINING_OUT / "logs",
        TRAINING_OUT / "curves",
        FIGURES_OUT,
        ROC_OUT,
        ARCHIVE_DATA,
    ):
        p.mkdir(parents=True, exist_ok=True)
    if on_eaf():
        for p in (SCRATCH_NPZ, SCRATCH_INFERENCE, SCRATCH_TRAINING):
            p.mkdir(parents=True, exist_ok=True)


def inference_pickle_name(stem: str, T: int, mode: str) -> str:
    return f"{stem}_T{T}_{mode}.pkl"


def resolve_inference_dirs(
    defect_type: str,
    model_type: str,
    mode: str,
    *,
    experiment: str = "ICARUS",
    plane: str = "plane1",
    base: Path | None = None,
) -> tuple[Path, Path]:
    """Return (nominal_dir, defect_dir) matching historical naming.

    Historical (scratch)::
        {base}/{plane}_healthy_outputs[-{mode}]{model_type}[/mixed_{mode}]
        {base}/{plane}_{defect}_outputs[-{mode}]{model_type}[/mixed_{mode}]

    Prefer ``base=SCRATCH_ICARUS`` on EAF for existing runs; new runs should use
    ``INFERENCE_OUT / experiment / ...``.
    """
    if base is None:
        base = SCRATCH_ICARUS if SCRATCH_ICARUS.exists() else INFERENCE_OUT / experiment

    if mode == "ddim2ddim":
        nominal = base / f"{plane}_healthy_outputs{model_type}"
        defect = base / f"{plane}_{defect_type}_outputs{model_type}"
    else:
        nominal = base / f"{plane}_healthy_outputs-{mode}{model_type}" / f"mixed_{mode}"
        defect = base / f"{plane}_{defect_type}_outputs-{mode}{model_type}" / f"mixed_{mode}"
    return nominal, defect


def default_model_checkpoint(prefer: str = "iterE") -> Path:
    """Default SBND diffusion checkpoint used in handscan / DDIM notebooks."""
    candidates = [
        GPUTNAM_TRAINING / prefer / "results" / "brats2update111000.pt",
        DATA_ROOT / "archive" / "ICARUS_NNs" / "nominal_best.pt",
        GPUTNAM_TRAINING / "iterE" / "results" / "brats2update111000.pt",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]


def describe_environment() -> dict:
    """Snapshot for the EAF probe notebook / debugging."""
    return {
        "hostname": hostname(),
        "on_eaf": on_eaf(),
        "app_root": str(APP_ROOT),
        "data_root": str(DATA_ROOT),
        "data_root_exists": DATA_ROOT.exists(),
        "scratch_root": str(SCRATCH_ROOT),
        "scratch_exists": SCRATCH_ROOT.exists(),
        "scratch_icarus_exists": SCRATCH_ICARUS.exists(),
        "diffusion_code_root": str(diffusion_code_root()),
        "default_checkpoint": str(default_model_checkpoint()),
        "default_checkpoint_exists": default_model_checkpoint().is_file(),
    }
