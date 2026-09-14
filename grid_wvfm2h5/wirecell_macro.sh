#!/bin/bash
# Endscript: dump waveforms to ROOT, then convert to g4-raw-*.h5 in the same job.

workdir=`pwd`
if [ -f celltree.log ]; then
        echo "The endscript has been executed!" | tee -a celltree.log
else
        touch celltree.log

    echo "################# Init of job" | tee -a celltree.log
	date | tee -a celltree.log

    echo "Contents of working directory" | tee -a celltree.log
    ls | tee -a celltree.log

    echo "Running root macro dump_waveform_all.C" | tee -a celltree.log
    root $INPUT_TAR_DIR_LOCAL_1/dump_waveform_all.C | tee -a celltree.log

    echo "Running wvfm2h5 conversion" | tee -a celltree.log
    if [ -f "$INPUT_TAR_DIR_LOCAL_1/wvfm2h5_run.sh" ]; then
      # Capture stderr too — compile failures were previously silent in celltree.log
      if ! bash "$INPUT_TAR_DIR_LOCAL_1/wvfm2h5_run.sh" waveform.root . 10 2>&1 | tee -a celltree.log; then
        echo "ERROR: wvfm2h5_run.sh failed (see above)" | tee -a celltree.log
      fi
    else
      echo "WARNING: wvfm2h5_run.sh not found in tarball" | tee -a celltree.log
    fi

    echo "Outputs after conversion:" | tee -a celltree.log
    ls -lh | tee -a celltree.log

    echo "################# Done!" | tee -a celltree.log

fi
