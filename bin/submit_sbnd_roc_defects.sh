#!/usr/bin/env bash
# SBND ROC: healthy + injected defects (bad_wire, coh_noise, charge_tail).
# 200 patches/class, 4 patches/job → -ngrid 50. T=200 ddim2ddim.
# Linear + anisotropic latest EMA (034000).
set -euo pipefail

REPO=/exp/sbnd/app/users/munjung/anomaly-detection
PNFS=/pnfs/sbnd/scratch/users/munjung/anomaly-detection
DATA=/exp/sbnd/data/users/munjung/anomaly-detection
CKPT_STEP=034000

export ANOMALY_WD="$REPO"
export ANOMALY_GRID_OUT_DIR="$PNFS/out"
export ANOMALY_USE_GIT_CLONE=0
export JOBSUB_MEMORY=6GB
export JOBSUB_DISK=20GB
export JOBSUB_LIFETIME=12h
export JOBSUB_CPU=4

cd "$REPO"

CLASSES=(healthy bad_wire coh_noise charge_tail)
CONFIGS=(linear anisotropic)

for cfg in "${CONFIGS[@]}"; do
  CKPT="$DATA/training/diffusion/${cfg}/emabrats2update_0.9999_${CKPT_STEP}.pt"
  test -f "$CKPT"
done

for probe in "${CLASSES[@]}"; do
  LIST="$PNFS/lists/roc_sbnd_${probe}_200.list"
  if [[ ! -f "$LIST" ]]; then
    echo "Missing $LIST — run: python samples/build_sbnd_roc_defect_dataset.py" >&2
    exit 1
  fi
  n=$(grep -cvE '^\s*(#|$)' "$LIST" || true)
  if [[ "$n" -ne 200 ]]; then
    echo "WARNING: $LIST has $n lines (expected 200)" >&2
  fi
done

for cfg in "${CONFIGS[@]}"; do
  CKPT="$DATA/training/diffusion/${cfg}/emabrats2update_0.9999_${CKPT_STEP}.pt"
  for probe in "${CLASSES[@]}"; do
    LIST="$PNFS/lists/roc_sbnd_${probe}_200.list"
    OUT="roc_sbnd_${probe}_${cfg}_ckpt${CKPT_STEP}_T200_ddim2ddim"
    echo "==== ${OUT} ===="
    python3 inference/submit_ddim_grid.py \
      -l "$LIST" \
      --model "$CKPT" \
      -o "$OUT" \
      -ngrid 50 \
      --T 200 \
      --batch-size 1 \
      --modes ddim2ddim \
      --extra-args "--train-config ${cfg}"
  done
done

echo DONE
