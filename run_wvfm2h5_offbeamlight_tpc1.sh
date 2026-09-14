#!/bin/bash
export BATCH_TAG="offbeamlight"
export TPC=1
export WAVEFORM_DIR="/pnfs/sbnd/scratch/users/munjung/v10_06_00/raw-SBND_DATA_InTimeCosmics_offbeamlight/waveforms"
export H5_DIR="/pnfs/sbnd/scratch/users/munjung/v10_06_00/raw-SBND_DATA_InTimeCosmics_offbeamlight/h5"
export USE_PARALLEL=0
export N_WORKERS=1
export PAUSE_EVERY=1
export PAUSE_COOLDOWN_SEC=30

exec /exp/sbnd/app/users/munjung/anomaly-detection/run_wvfm2h5_batch.sh
