#!/usr/bin/env bash
# Sync a subset of SBND training MC NPZs from EAF scratch → durable /exp (+ pnfs).
#
# Must run on an EAF node where /scratch/7Day* is mounted (e.g. jupyter-munjung).
# After this, sbndbuild can rebuild ROC samples from samples/npz/healthy.
#
# Usage:
#   bash bin/sync_eaf_training_npz.sh
#   N_COPY=500 bash bin/sync_eaf_training_npz.sh
set -euo pipefail

USER_NAME="${USER:-munjung}"
DATA_ROOT="${ANOMALY_DATA_ROOT:-/exp/sbnd/data/users/${USER_NAME}/anomaly-detection}"
PNFS_ROOT="${ANOMALY_PNFS_ROOT:-/pnfs/sbnd/scratch/users/${USER_NAME}/anomaly-detection}"
N_COPY="${N_COPY:-500}"

SCRATCH_NPZ=""
for pool in /scratch/7DayLifetime /scratch/7DayScratch /scratch/7Day; do
  for cand in \
      "${pool}/${USER_NAME}/anomaly-detection/npz/healthy" \
      "${pool}/${USER_NAME}/anomaly-detection/npz"; do
    if [[ -d "$cand" ]] && compgen -G "${cand}/*.npz" >/dev/null; then
      SCRATCH_NPZ="$cand"
      break 2
    fi
  done
done

if [[ -z "$SCRATCH_NPZ" ]]; then
  echo "ERROR: training NPZs not found under /scratch/7Day*/${USER_NAME}/anomaly-detection/npz" >&2
  echo "hostname=$(hostname)" >&2
  ls -d /scratch/7Day* 2>/dev/null || true
  exit 1
fi

DEST="${DATA_ROOT}/samples/npz/healthy"
PNFS_DEST="${PNFS_ROOT}/inputs/training_mc_healthy"
mkdir -p "$DEST" "$PNFS_DEST"

echo "hostname=$(hostname)"
echo "SRC=$SCRATCH_NPZ"
echo "DEST=$DEST"
echo "N_COPY=$N_COPY"

mapfile -t all < <(ls -1 "$SCRATCH_NPZ"/*.npz 2>/dev/null | sort)
echo "available=${#all[@]}"
if [[ ${#all[@]} -eq 0 ]]; then
  echo "ERROR: no .npz in $SCRATCH_NPZ" >&2
  exit 1
fi

mapfile -t prefer < <(printf '%s\n' "${all[@]}" | grep -E 'tpc[0-9]_plane[0-9]_rec_' || true)
if [[ ${#prefer[@]} -gt 0 ]]; then
  pool=("${prefer[@]}")
else
  pool=("${all[@]}")
fi

# Deterministic head of sorted list (stable across hosts)
mapfile -t chosen < <(printf '%s\n' "${pool[@]}" | head -n "$N_COPY")

copied=0
for f in "${chosen[@]}"; do
  [[ -f "$f" ]] || continue
  base="$(basename "$f")"
  cp -f "$f" "${DEST}/${base}"
  cp -f "$f" "${PNFS_DEST}/${base}"
  copied=$((copied + 1))
done

echo "DONE copied=${copied} → ${DEST}"
ls "$DEST" | wc -l
