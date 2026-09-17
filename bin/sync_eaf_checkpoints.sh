#!/usr/bin/env bash
# Sync representative EMA training checkpoints from EAF scratch → durable /exp + pnfs.
#
# Must run on an EAF node where /scratch/7Day* is mounted (e.g. jupyter-munjung).
# Shared /exp and /pnfs are visible from sbndbuild / grid afterwards.
#
# Usage:
#   bash bin/sync_eaf_checkpoints.sh
#   STEPS="10000 20000 23000" CONFIGS="linear anisotropic" bash bin/sync_eaf_checkpoints.sh
set -euo pipefail

USER_NAME="${USER:-munjung}"
APP_ROOT="${ANOMALY_WD:-/exp/sbnd/app/users/munjung/anomaly-detection}"
DATA_ROOT="${ANOMALY_DATA_ROOT:-/exp/sbnd/data/users/munjung/anomaly-detection}"
PNFS_ROOT="${ANOMALY_PNFS_ROOT:-/pnfs/sbnd/scratch/users/${USER_NAME}/anomaly-detection}"

# Discover scratch run root
SCRATCH_RUN=""
for pool in /scratch/7DayLifetime /scratch/7DayScratch /scratch/7Day; do
  cand="${pool}/${USER_NAME}/anomaly-detection/training/diffusion"
  if [[ -d "$cand" ]]; then
    SCRATCH_RUN="$cand"
    break
  fi
done
if [[ -z "$SCRATCH_RUN" ]]; then
  echo "ERROR: EAF training scratch not found under /scratch/7Day*/${USER_NAME}/anomaly-detection/training/diffusion" >&2
  echo "hostname=$(hostname)  scratch pools:" >&2
  ls -d /scratch/7Day* 2>/dev/null || true
  exit 1
fi

CONFIGS="${CONFIGS:-linear anisotropic}"

EXP_OUT="${DATA_ROOT}/training/diffusion"
PNFS_OUT="${PNFS_ROOT}/models_training"
mkdir -p "$EXP_OUT" "$PNFS_OUT"

echo "hostname=$(hostname)"
echo "SCRATCH_RUN=$SCRATCH_RUN"
echo "EXP_OUT=$EXP_OUT"
echo "PNFS_OUT=$PNFS_OUT"
echo "CONFIGS=$CONFIGS"

copied=0
for cfg in $CONFIGS; do
  src_dir="${SCRATCH_RUN}/${cfg}"
  if [[ ! -d "$src_dir" ]]; then
    echo "SKIP missing config dir: $src_dir"
    continue
  fi
  mkdir -p "${EXP_OUT}/${cfg}" "${PNFS_OUT}/${cfg}"
  mapfile -t emas < <(ls -1 "${src_dir}"/emabrats2update_0.9999_*.pt 2>/dev/null | sort || true)
  if [[ ${#emas[@]} -eq 0 ]]; then
    echo "SKIP no EMA ckpts in $src_dir"
    continue
  fi
  echo "=== ${cfg}: ${#emas[@]} EMA ckpts (showing last 8) ==="
  printf '  %s\n' "${emas[@]: -8}"

  # Prefer 10k, 20k, and newest available
  declare -A want=()
  for f in "${emas[@]}"; do
    base="$(basename "$f")"
    step="${base##*_}"
    step="${step%.pt}"
    step=$((10#$step))
    if [[ "$step" -eq 10000 || "$step" -eq 20000 ]]; then
      want["$base"]="$f"
    fi
  done
  newest="${emas[-1]}"
  want["$(basename "$newest")"]="$newest"
  # If fewer than 2 preferred hits, also take earliest >=10k
  if [[ ${#want[@]} -lt 2 ]]; then
    for f in "${emas[@]}"; do
      base="$(basename "$f")"
      step="${base##*_}"; step="${step%.pt}"; step=$((10#$step))
      if [[ "$step" -ge 10000 ]]; then
        want["$base"]="$f"
        break
      fi
    done
  fi

  for base in "${!want[@]}"; do
    f="${want[$base]}"
    echo "copy $f"
    cp -f "$f" "${EXP_OUT}/${cfg}/${base}"
    cp -f "$f" "${PNFS_OUT}/${cfg}/${base}"
    copied=$((copied + 1))
    ls -lh "${EXP_OUT}/${cfg}/${base}" "${PNFS_OUT}/${cfg}/${base}"
  done
  # Keep the longer learning-curve history (never replace durable with a shorter scratch copy).
  src_prog="${src_dir}/progress.csv"
  dst_prog="${EXP_OUT}/${cfg}/progress.csv"
  if [[ -f "$src_prog" ]]; then
    src_n=$(wc -l < "$src_prog" | tr -d ' ')
    dst_n=0
    if [[ -f "$dst_prog" ]]; then
      dst_n=$(wc -l < "$dst_prog" | tr -d ' ')
    fi
    if [[ "$src_n" -ge "$dst_n" ]]; then
      cp -f "$src_prog" "$dst_prog"
      echo "progress.csv synced (${src_n} lines) -> $dst_prog"
    else
      echo "progress.csv KEEP durable (${dst_n} lines) > scratch (${src_n} lines)"
    fi
  fi
done

echo "DONE copied=${copied} objects"
echo "List EXP:"
find "$EXP_OUT" -name 'emabrats*.pt' | sort
echo "List PNFS:"
find "$PNFS_OUT" -name 'emabrats*.pt' | sort
