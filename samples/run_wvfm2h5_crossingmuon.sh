#!/bin/bash
BASE="/pnfs/sbnd/scratch/users/munjung/v10_06_00/data_MCP2025B_CrossingMuon_FullRun1_AfterLight_8_crossingmuon"
export BATCH_TAG="crossingmuon"
export WAVEFORM_DIR="$BASE/waveforms"
export H5_DIR="$BASE/h5"
export USE_PARALLEL=0
export N_WORKERS=1
export PAUSE_EVERY=1
export PAUSE_COOLDOWN_SEC=30
export TPC="${TPC:-0}"

exec /exp/sbnd/app/users/munjung/anomaly-detection/run_wvfm2h5_batch.sh
