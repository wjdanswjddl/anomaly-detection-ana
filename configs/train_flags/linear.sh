#!/usr/bin/env bash
# Config: linear (nominal) — standard Ho et al. β schedule, ε-prediction.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

export CONFIG_NAME="linear"
export DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --predict_xstart False --anisotropic_noise False"
export IMAGE_TRAIN_FLAGS="$DATADIR $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS"
