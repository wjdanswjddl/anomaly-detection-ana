#!/bin/bash
# Create pnfs directories and copy Gray iterE EMA checkpoint for grid inference.
set -euo pipefail
USER_NAME="${USER:-munjung}"
ROOT="/pnfs/sbnd/scratch/users/${USER_NAME}/anomaly-detection"
MODEL_SRC="${MODEL_SRC:-/exp/sbnd/data/users/gputnam/training-SBND/iterE/results/emabrats2update_0.9999_111000.pt}"

mkdir -p "${ROOT}/models" "${ROOT}/lists" "${ROOT}/out"
echo "ROOT=${ROOT}"

if [ ! -f "${MODEL_SRC}" ]; then
  echo "Missing model source: ${MODEL_SRC}" >&2
  exit 1
fi

dest="${ROOT}/models/$(basename "${MODEL_SRC}")"
if [ -f "${dest}" ]; then
  echo "Model already present: ${dest}"
else
  echo "Copying model -> ${dest}"
  cp -f "${MODEL_SRC}" "${dest}"
fi
ls -lh "${dest}"
echo "Done. Put NPZ paths in ${ROOT}/lists/*.list"
