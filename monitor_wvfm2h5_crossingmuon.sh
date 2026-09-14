#!/bin/bash
BASE="/pnfs/sbnd/scratch/users/munjung/v10_06_00/data_MCP2025B_CrossingMuon_FullRun1_AfterLight_8_crossingmuon"
export BATCH_TAG="crossingmuon"
export WAVEFORM_DIR="$BASE/waveforms"
export H5_DIR="$BASE/h5"
export BATCH_SCRIPT="/exp/sbnd/app/users/munjung/anomaly-detection/run_wvfm2h5_crossingmuon_both.sh"
export JOB_MEM_ONLY=1
export JOB_MEM_MAX_GB=20
# Use tpc0 log namespace for the combined both-TPC runner.
export TPC=0

exec /exp/sbnd/app/users/munjung/anomaly-detection/monitor_wvfm2h5_batch.sh
