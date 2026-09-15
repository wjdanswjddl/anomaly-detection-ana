"""Registry of the five diffusion training configurations.

Matches ICARUS ``noise_variations`` runs and ``archive/ICARUS_NNs/*_best.pt``:
linear, cosine, ramp, anisotropic, pred_xstart.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
FLAGS_DIR = APP_ROOT / "configs" / "train_flags"
IMPORTED_FLAGS_DIR = FLAGS_DIR / "imported"


@dataclass(frozen=True)
class TrainConfig:
    name: str
    flag_script: Path
    """Directory-name suffix used historically in inference outputs ("" for linear)."""
    model_type_suffix: str
    description: str
    noise_schedule: str
    predict_xstart: bool
    anisotropic_noise: bool


# Order matches CompareReconstructions-ICARUS model_dict.
TRAIN_CONFIGS: tuple[TrainConfig, ...] = (
    TrainConfig(
        name="linear",
        flag_script=FLAGS_DIR / "linear.sh",
        model_type_suffix="",
        description="Linear β schedule, ε-prediction (nominal)",
        noise_schedule="linear",
        predict_xstart=False,
        anisotropic_noise=False,
    ),
    TrainConfig(
        name="cosine",
        flag_script=FLAGS_DIR / "cosine.sh",
        model_type_suffix="-cosine",
        description="Cosine β schedule, ε-prediction",
        noise_schedule="cosine",
        predict_xstart=False,
        anisotropic_noise=False,
    ),
    TrainConfig(
        name="ramp",
        flag_script=FLAGS_DIR / "ramp.sh",
        model_type_suffix="-ramp",
        description="Ramp (piecewise) β schedule, ε-prediction",
        noise_schedule="ramp",
        predict_xstart=False,
        anisotropic_noise=False,
    ),
    TrainConfig(
        name="anisotropic",
        flag_script=FLAGS_DIR / "anisotropic.sh",
        model_type_suffix="-anisotropic",
        description="Linear β + signal-proportional anisotropic noise",
        noise_schedule="linear",
        predict_xstart=False,
        anisotropic_noise=True,
    ),
    TrainConfig(
        name="pred_xstart",
        flag_script=FLAGS_DIR / "pred_xstart.sh",
        model_type_suffix="-pred_xstart",
        description="Linear β, predict x0 instead of ε",
        noise_schedule="linear",
        predict_xstart=True,
        anisotropic_noise=False,
    ),
)

TRAIN_CONFIG_BY_NAME = {c.name: c for c in TRAIN_CONFIGS}


def resolve_flag_script(name: str) -> Path:
    """Prefer imported EAF flags when present, else local bundle."""
    cfg = TRAIN_CONFIG_BY_NAME[name]
    imported = IMPORTED_FLAGS_DIR / cfg.flag_script.name
    if imported.is_file():
        return imported
    # Also accept imported names like model_flags_SBND_cosine.sh
    for alt in IMPORTED_FLAGS_DIR.glob(f"*{name}*.sh"):
        return alt
    return cfg.flag_script


def list_configs_table() -> str:
    lines = ["name | schedule | x0 | aniso | flag", "-----|----------|----|-------|-----"]
    for c in TRAIN_CONFIGS:
        flag = resolve_flag_script(c.name)
        lines.append(
            f"{c.name} | {c.noise_schedule} | {c.predict_xstart} | "
            f"{c.anisotropic_noise} | {flag}"
        )
    return "\n".join(lines)
