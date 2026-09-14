#!/bin/bash
export BATCH_TAG="offbeamlight"
export TPC=1
export WAVEFORM_DIR="/pnfs/sbnd/scratch/users/munjung/v10_06_00/raw-SBND_DATA_InTimeCosmics_offbeamlight/waveforms"
export H5_DIR="/pnfs/sbnd/scratch/users/munjung/v10_06_00/raw-SBND_DATA_InTimeCosmics_offbeamlight/h5"
export BATCH_SCRIPT="/exp/sbnd/app/users/munjung/anomaly-detection/run_wvfm2h5_offbeamlight_tpc1.sh"

exec /exp/sbnd/app/users/munjung/anomaly-detection/monitor_wvfm2h5_batch.sh
