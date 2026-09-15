#!/usr/bin/env bash
# Config: anisotropic — linear β + signal-proportional spatial noise (DiffSSC-style).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"

export CONFIG_NAME="anisotropic"
export DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --predict_xstart False --anisotropic_noise True --noise_mode signal_proportional --empty_noise_fraction 0.05 --smoothing_sigma 2.0"
export IMAGE_TRAIN_FLAGS="$DATADIR $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS"
