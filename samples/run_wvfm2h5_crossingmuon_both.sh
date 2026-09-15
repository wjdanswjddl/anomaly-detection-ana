#!/bin/bash
# Run crossing-muon conversion for TPC 0 then TPC 1 (g4-raw-0 / g4-raw-1).
set -uo pipefail

BASE="/pnfs/sbnd/scratch/users/munjung/v10_06_00/data_MCP2025B_CrossingMuon_FullRun1_AfterLight_8_crossingmuon"
SCRIPT="/exp/sbnd/app/users/munjung/anomaly-detection/run_wvfm2h5_crossingmuon.sh"
LOG_DIR="/exp/sbnd/data/users/munjung/anomaly-detection/logs"
mkdir -p "$LOG_DIR" "$BASE/h5"

# Clear stalled partial for the interrupted TPC0 file so it is retried cleanly.
PARTIAL="$BASE/h5/71789286_139"
if [ -d "$PARTIAL" ] && [ ! -f "$PARTIAL/.tpc0.done" ]; then
    echo "Clearing partial TPC0 outputs in $PARTIAL"
    rm -f "$PARTIAL"/g4-raw-0_*.h5
fi

for tpc in 0 1; do
    echo "=== Starting crossingmuon TPC $tpc at $(date) ==="
    export TPC="$tpc"
    "$SCRIPT"
    echo "=== Finished crossingmuon TPC $tpc at $(date) ==="
done
