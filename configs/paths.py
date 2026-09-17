"""Unified paths and naming for the anomaly-detection workflow.

Code lives under APP_ROOT (this repo). Large products and outputs live under
DATA_ROOT (parallel /exp/sbnd/data tree). GPU training / bulk inference on EAF
uses a scratch root that is **discovered at runtime** (never assumed).

Import from notebooks/scripts::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(".../anomaly-detection")))  # repo root
    from configs.paths import APP_ROOT, DATA_ROOT, ensure_layout, resolve_inference_dirs
"""
from __future__ import annotations

import os
import shutil
import socket
from pathlib import Path

# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------

APP_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/exp/sbnd/data/users/munjung/anomaly-detection")

# Shared checkpoints from gputnam (read-only on /exp)
GPUTNAM_TRAINING = Path("/exp/sbnd/data/users/gputnam/training-SBND")
GPUTNAM_DNN_ROI = Path("/exp/sbnd/data/users/gputnam/DNN-ROI-images")

# Upstream SBND VAE/CAE (durable copy under /exp; scratch iterA often purged)
GPUTNAM_AE_SBND_ROOT = Path("/exp/sbnd/data/users/gputnam/diffusion-anomaly-SBND")
GPUTNAM_VAE_SBND_SCRATCH = GPUTNAM_AE_SBND_ROOT / "VAE-iterA"
GPUTNAM_CAE_SBND_SCRATCH = GPUTNAM_AE_SBND_ROOT / "CAE-iterA"
GPUTNAM_VAE_SBND_CKPT = "ema_0.9999_263000.pt"
GPUTNAM_CAE_SBND_CKPT = "ema_0.9999_263000.pt"

# Vendored / cloned diffusion training code (prefer EAF clone when discovered)
DIFFUSION_ANOMALY_APP = APP_ROOT / "train" / "diffusion-anomaly"
DIFFUSION_ANOMALY_SIBLING = Path("/exp/sbnd/app/users/munjung/diffusion-anomaly")

# ---------------------------------------------------------------------------
# Scratch discovery — probe /scratch; never invent a 7Day* name
# ---------------------------------------------------------------------------

_SCRATCH_TOP = Path("/scratch")


def existing_scratch_pools() -> list[Path]:
    """Return directories that actually exist under ``/scratch`` (e.g. 7Day*)."""
    if not _SCRATCH_TOP.is_dir():
        return []
    pools: list[Path] = []
    try:
        for p in sorted(_SCRATCH_TOP.iterdir()):
            if p.is_dir() and p.name.startswith("7Day"):
                pools.append(p)
    except OSError:
        return []
    return pools


def discover_scratch_root(
    *,
    user: str = "munjung",
    project: str = "anomaly-detection",
) -> Path | None:
    """Find an existing EAF scratch project directory.

    Order:
      1. ``ANOMALY_SCRATCH_ROOT`` if set **and that path is an existing directory**
      2. ``/scratch/<existing-7Day-pool>/<user>/<project>`` if that path exists
      3. ``/scratch/<existing-7Day-pool>/<user>/<project>`` only if
         ``/scratch/<pool>/<user>`` already exists (project may be created later)

    Returns None if no usable scratch is present (e.g. on sbndbuild*).
    """
    env = os.environ.get("ANOMALY_SCRATCH_ROOT")
    if env:
        env_path = Path(env).expanduser()
        if env_path.is_dir():
            return env_path

    pools = existing_scratch_pools()
    # Prefer a pool that already has the project tree.
    for pool in pools:
        project_dir = pool / user / project
        if project_dir.is_dir():
            return project_dir
    # Next: user dir exists — project can be created under it.
    for pool in pools:
        user_dir = pool / user
        if user_dir.is_dir():
            return user_dir / project
    return None


def require_existing_dir(path: Path, label: str) -> Path:
    """Raise FileNotFoundError unless ``path`` is an existing directory."""
    if not path.is_dir():
        raise FileNotFoundError(
            f"{label} does not exist: {path}. "
            f"Existing /scratch/7Day* pools: "
            f"{[str(p) for p in existing_scratch_pools()] or ['<none on this host>']}."
        )
    return path


def ensure_under_existing_parent(path: Path, label: str) -> Path:
    """Create ``path`` only if its parent already exists; else raise."""
    parent = path.parent
    if not parent.is_dir():
        raise FileNotFoundError(
            f"Cannot create {label}={path}: parent missing ({parent}). "
            f"Existing /scratch/7Day* pools: "
            f"{[str(p) for p in existing_scratch_pools()] or ['<none on this host>']}."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


# Resolved at import from whatever actually exists on this host.
SCRATCH_ROOT: Path | None = discover_scratch_root()

# ---------------------------------------------------------------------------
# Data-area layout (outputs)
# ---------------------------------------------------------------------------

SAMPLES_H5 = DATA_ROOT / "samples" / "h5"
SAMPLES_NPZ = DATA_ROOT / "samples" / "npz"
SAMPLES_HANDSCAN = DATA_ROOT / "samples" / "handscan"
SAMPLES_DEFECTS = DATA_ROOT / "samples" / "defects"
HANDSCAN_VALIDATION = SAMPLES_HANDSCAN / "validation"
# Patch-curated handscan datasets (from CurateHandscanPatches.ipynb)
HANDSCAN_CURATED = SAMPLES_HANDSCAN / "curated"

INFERENCE_OUT = DATA_ROOT / "inference"
TRAINING_OUT = DATA_ROOT / "training"
FIGURES_OUT = DATA_ROOT / "figures"
ROC_OUT = DATA_ROOT / "roc_curves"
ARCHIVE_DATA = DATA_ROOT / "archive"
HANDSCAN_ARCHIVES = ARCHIVE_DATA / "handscan_archives"
# Auto full-plane crops without patch curation (relocated; do not submit)
HANDSCAN_GRID_AUTO_ARCHIVED = ARCHIVE_DATA / "handscan_grid_auto_uncurated"

# Scratch layout — None when no EAF scratch pool is mounted on this host.
SCRATCH_NPZ = (SCRATCH_ROOT / "npz") if SCRATCH_ROOT is not None else None
SCRATCH_INFERENCE = (SCRATCH_ROOT / "inference") if SCRATCH_ROOT is not None else None
SCRATCH_TRAINING = (SCRATCH_ROOT / "training") if SCRATCH_ROOT is not None else None

SCRATCH_ICARUS: Path | None = None
for _pool in existing_scratch_pools():
    _cand = _pool / "munjung" / "ICARUS"
    if _cand.is_dir():
        SCRATCH_ICARUS = _cand
        break

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
    """True when an EAF 7Day scratch pool is mounted (or diffusion conda exists)."""
    return bool(existing_scratch_pools()) or Path("/home/munjung/.conda/envs/diffusion").exists()


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


def resolve_training_run_root() -> Path:
    """Directory for training logs/checkpoints.

    Prefers discovered scratch ``.../training/diffusion`` when the scratch
    project root can be created under an existing user dir; otherwise uses
    durable ``DATA_ROOT/training/diffusion``.
    """
    if SCRATCH_ROOT is not None:
        # Ensure project root only if its parent (user dir) exists.
        ensure_under_existing_parent(SCRATCH_ROOT, "SCRATCH_ROOT")
        run_root = SCRATCH_ROOT / "training" / "diffusion"
        ensure_under_existing_parent(run_root, "RUN_ROOT")
        return run_root
    run_root = TRAINING_OUT / "diffusion"
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def _progress_csv_nrows(path: Path) -> int:
    """Data rows in a progress.csv (excludes header). Missing/empty → -1."""
    if not path.is_file() or path.stat().st_size == 0:
        return -1
    with path.open() as f:
        n = sum(1 for _ in f) - 1
    return max(n, 0)


def seed_progress_csv(log_dir: Path, config_name: str) -> Path | None:
    """Ensure ``log_dir/progress.csv`` keeps the longest known history.

    Scratch pools expire; durable copies live under ``DATA_ROOT``. Before a
    resume, copy the longest ``progress.csv`` into the active log dir so the
    learning curve continues instead of restarting from an empty/truncated file.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    dest = log_dir / "progress.csv"
    candidates = [
        dest,
        TRAINING_OUT / "diffusion" / config_name / "progress.csv",
    ]
    best: Path | None = None
    best_n = -1
    for p in candidates:
        n = _progress_csv_nrows(p)
        if n > best_n:
            best_n = n
            best = p
    if best is None or best_n < 0:
        return None
    if best.resolve() != dest.resolve():
        shutil.copy2(best, dest)
    return dest


def resolve_ae_training_run_root(ae_type: str, experiment: str = "sbnd") -> Path:
    """Log/checkpoint root for VAE/CAE training: ``.../training/{vae|cae}/{experiment}``."""
    ae = ae_type.lower()
    if ae not in ("vae", "cae"):
        raise ValueError(f"ae_type must be vae|cae, got {ae_type!r}")
    exp = experiment.lower()
    if SCRATCH_ROOT is not None:
        # Create .../training/{ae}/{exp} one level at a time under an existing
        # scratch project root (ensure_under_existing_parent is not recursive).
        ensure_under_existing_parent(SCRATCH_ROOT, "SCRATCH_ROOT")
        training = ensure_under_existing_parent(SCRATCH_ROOT / "training", "SCRATCH_TRAINING")
        ae_dir = ensure_under_existing_parent(training / ae, f"AE_{ae}_ROOT")
        return ensure_under_existing_parent(ae_dir / exp, "AE_RUN_ROOT")
    run_root = TRAINING_OUT / ae / exp
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def default_ae_checkpoint(ae_type: str, experiment: str = "sbnd") -> Path | None:
    """Best available EMA checkpoint for a VAE/CAE baseline.

    Search order (first existing file wins):
      1. ``DATA_ROOT/training/{ae}/{experiment}/`` EMA / model checkpoints
      2. Gputnam durable SBND iterA EMA (``GPUTNAM_*_SBND_SCRATCH``)
      3. Scratch training tree under ``SCRATCH_ROOT``
    """
    ae = ae_type.lower()
    exp = experiment.lower()
    durable = TRAINING_OUT / ae / exp
    scratch_known = {
        ("vae", "sbnd"): GPUTNAM_VAE_SBND_SCRATCH / GPUTNAM_VAE_SBND_CKPT,
        ("cae", "sbnd"): GPUTNAM_CAE_SBND_SCRATCH / GPUTNAM_CAE_SBND_CKPT,
    }
    candidates: list[Path] = []
    if durable.is_dir():
        candidates.extend(sorted(durable.glob("ema_*.pt"), reverse=True))
        candidates.extend(sorted(durable.glob("emabrats2update_*.pt"), reverse=True))
        candidates.extend(sorted(durable.glob("model*.pt"), reverse=True))
        candidates.extend(sorted(durable.glob("brats2update*.pt"), reverse=True))
    known = scratch_known.get((ae, exp))
    if known is not None:
        candidates.append(known)
    if SCRATCH_ROOT is not None:
        scratch_dir = SCRATCH_ROOT / "training" / ae / exp
        if scratch_dir.is_dir():
            candidates.extend(sorted(scratch_dir.glob("ema_*.pt"), reverse=True))
            candidates.extend(sorted(scratch_dir.glob("emabrats2update_*.pt"), reverse=True))
            candidates.extend(sorted(scratch_dir.glob("brats2update*.pt"), reverse=True))
    for c in candidates:
        if c.is_file():
            return c
    return None


def ae_model_config(ae_type: str) -> dict:
    """Hard-coded configs matching model_flags_{VAE,CAE}_*.sh / demo scripts."""
    ae = ae_type.lower()
    common = dict(
        image_size=512,
        in_channels=1,
        ae_hidden_dims="32,64,128,256,512,512,512",
        ae_latent_dim=512,
        final_activation="tanh",
    )
    if ae == "vae":
        return dict(
            ae_type="vae",
            kld_weight=1e-4,
            spatial_latent=True,  # unused by VAE
            ssim_weight=0.0,      # unused by VAE
            **common,
        )
    if ae == "cae":
        return dict(
            ae_type="cae",
            kld_weight=1e-4,      # unused by CAE
            spatial_latent=True,
            ssim_weight=0.0,
            **common,
        )
    raise ValueError(f"ae_type must be vae|cae, got {ae_type!r}")


def resolve_training_data_dir(discover: dict | None = None) -> Path:
    """NPZ training data directory — must already exist."""
    discover = discover or {}
    candidates: list[Path] = []
    hint = discover.get("scratch_anomaly")
    if hint:
        candidates.append(Path(hint) / "npz")
    if SCRATCH_NPZ is not None:
        candidates.append(SCRATCH_NPZ)
    candidates.append(SAMPLES_NPZ)
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(
        "No training NPZ directory found. Looked at: "
        + ", ".join(str(c) for c in candidates)
        + f". Existing /scratch/7Day* pools: "
        + f"{[str(p) for p in existing_scratch_pools()] or ['<none on this host>']}."
    )


def ensure_layout() -> None:
    """Create expected output directories under DATA_ROOT (and scratch if present)."""
    for p in (
        SAMPLES_H5,
        SAMPLES_NPZ,
        SAMPLES_HANDSCAN,
        SAMPLES_DEFECTS,
        HANDSCAN_VALIDATION,
        HANDSCAN_CURATED,
        INFERENCE_OUT,
        TRAINING_OUT / "checkpoints",
        TRAINING_OUT / "logs",
        TRAINING_OUT / "curves",
        TRAINING_OUT / "vae",
        TRAINING_OUT / "cae",
        FIGURES_OUT,
        ROC_OUT,
        ARCHIVE_DATA,
        HANDSCAN_ARCHIVES,
    ):
        p.mkdir(parents=True, exist_ok=True)
    # Only mkdir scratch children when the scratch project root already exists.
    if SCRATCH_ROOT is not None and SCRATCH_ROOT.is_dir():
        for p in (SCRATCH_NPZ, SCRATCH_INFERENCE, SCRATCH_TRAINING):
            if p is not None:
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
        base = SCRATCH_ICARUS if (SCRATCH_ICARUS is not None and SCRATCH_ICARUS.exists()) else INFERENCE_OUT / experiment

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
        "scratch_pools": [str(p) for p in existing_scratch_pools()],
        "scratch_root": str(SCRATCH_ROOT) if SCRATCH_ROOT is not None else None,
        "scratch_exists": bool(SCRATCH_ROOT is not None and SCRATCH_ROOT.is_dir()),
        "scratch_icarus": str(SCRATCH_ICARUS) if SCRATCH_ICARUS is not None else None,
        "scratch_icarus_exists": bool(SCRATCH_ICARUS is not None and SCRATCH_ICARUS.is_dir()),
        "diffusion_code_root": str(diffusion_code_root()),
        "default_checkpoint": str(default_model_checkpoint()),
        "default_checkpoint_exists": default_model_checkpoint().is_file(),
        "vae_sbnd_checkpoint": str(default_ae_checkpoint("vae", "sbnd")),
        "cae_sbnd_checkpoint": str(default_ae_checkpoint("cae", "sbnd")),
        "vae_icarus_checkpoint": str(default_ae_checkpoint("vae", "icarus")),
        "cae_icarus_checkpoint": str(default_ae_checkpoint("cae", "icarus")),
        "gputnam_dnn_roi": str(GPUTNAM_DNN_ROI),
        "gputnam_dnn_roi_exists": GPUTNAM_DNN_ROI.is_dir(),
    }
