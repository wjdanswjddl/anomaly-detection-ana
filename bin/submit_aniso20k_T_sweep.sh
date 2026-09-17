#!/usr/bin/env bash
# T-sweep stop-signal jobs on best training ckpt (anisotropic @ 20k by AUC).
set -euo pipefail

REPO=/exp/sbnd/app/users/munjung/anomaly-detection
CKPT=/exp/sbnd/data/users/munjung/anomaly-detection/training/diffusion/anisotropic/emabrats2update_0.9999_020000.pt
PNFS=/pnfs/sbnd/scratch/users/munjung/anomaly-detection
TS=(200 400 600 800 1000)
T_TAG=$(IFS=-; echo "${TS[*]}")

export ANOMALY_WD="$REPO"
export ANOMALY_GRID_OUT_DIR="$PNFS/out"
export ANOMALY_USE_GIT_CLONE=0
export JOBSUB_MEMORY=8GB
export JOBSUB_DISK=30GB
export JOBSUB_LIFETIME=24h
export JOBSUB_CPU=4

cd "$REPO"
test -f "$CKPT"

submit_one () {
  local probe="$1"
  local list="$2"
  local out
  if [[ "$probe" == "coh" ]]; then
    out="stopsignal_anisotropic_ckpt020000_T${T_TAG}"
  else
    out="stopsignal_${probe}_anisotropic_ckpt020000_T${T_TAG}"
  fi
  echo "==== ${out} ===="
  python3 inference/submit_ddim_grid.py \
    -l "$list" \
    --model "$CKPT" \
    -o "$out" \
    -ngrid 12 \
    --T "${TS[@]}" \
    --batch-size 1 \
    --extra-args "--train-config anisotropic"
}

submit_one healthy "$PNFS/lists/stop_signal_healthy_patches.list"
submit_one coh     "$PNFS/lists/stop_signal_coh_noise_patches.list"
echo DONE
