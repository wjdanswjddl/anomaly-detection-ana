#!/usr/bin/env bash
# Submit stop-signal healthy + coh probes with Gray iterE EMA (111k).
set -euo pipefail

REPO=/exp/sbnd/app/users/munjung/anomaly-detection
MODEL=/pnfs/sbnd/scratch/users/munjung/anomaly-detection/models/emabrats2update_0.9999_111000.pt
PNFS=/pnfs/sbnd/scratch/users/munjung/anomaly-detection

export ANOMALY_WD="$REPO"
export ANOMALY_GRID_OUT_DIR="$PNFS/out"
export ANOMALY_USE_GIT_CLONE=0
export JOBSUB_MEMORY=6GB
export JOBSUB_DISK=20GB
export JOBSUB_LIFETIME=12h
export JOBSUB_CPU=4

cd "$REPO"

echo "==== healthy / Gray 111k T=200 ===="
python3 inference/submit_ddim_grid.py \
  -l "$PNFS/lists/stop_signal_healthy_patches.list" \
  --model "$MODEL" \
  -o stopsignal_healthy_gray_ckpt111000_T200 \
  -ngrid 12 \
  --T 200 \
  --batch-size 1 \
  --extra-args "--train-config linear"

echo "==== coh / Gray 111k T=200 ===="
python3 inference/submit_ddim_grid.py \
  -l "$PNFS/lists/stop_signal_coh_noise_patches.list" \
  --model "$MODEL" \
  -o stopsignal_gray_ckpt111000_T200 \
  -ngrid 12 \
  --T 200 \
  --batch-size 1 \
  --extra-args "--train-config linear"

echo DONE
