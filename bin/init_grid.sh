#!/bin/bash
# Bootstrap Python venv on an SBND grid worker for anomaly-detection inference.
# Sourced from bin/grid_executable.sh after the repo is cloned.
set -euo pipefail

echo "BEARER_TOKEN_FILE=${BEARER_TOKEN_FILE:-<unset>}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p envs
cd envs
VENV_NAME="${ANOMALY_VENV_NAME:-venv_py310_anomaly}"

if [ -d "${VENV_NAME}" ]; then
  echo "[init_grid] reusing venv ${VENV_NAME}"
else
  echo "[init_grid] creating venv ${VENV_NAME}"
  python3 -m venv "${VENV_NAME}"
fi

# shellcheck disable=SC1090
source "${VENV_NAME}/bin/activate"

python -m pip install --upgrade pip wheel setuptools

echo "[init_grid] installing numpy tqdm blobfile"
python -m pip install "numpy>=1.24" "tqdm>=4.66" "blobfile>=2.0"

echo "[init_grid] installing CPU torch"
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch

cd "${REPO_ROOT}"
export ANOMALY_WD="${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/_stubs:${REPO_ROOT}/train/diffusion-anomaly:${REPO_ROOT}/inference:${REPO_ROOT}:${PYTHONPATH:-}"
echo "[init_grid] ANOMALY_WD=${ANOMALY_WD}"
echo "[init_grid] PYTHONPATH=${PYTHONPATH}"
python -c "import torch, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'numpy', numpy.__version__)"
