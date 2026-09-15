#!/bin/bash
# Condor / jobsub worker entrypoint for DDIM inference (NPZ in, pickle/npz out).
#
# Args from jobsub_submit trailing arguments:
#   $1 = pnfs output directory (ifdh target)
#   $2 = output name prefix (used in log filenames)
#
# Per-process script run_${PROCESS}.sh is shipped in the dropbox tarball and stages
# input *.npz files + the model checkpoint, then runs run_ddim2ddim_inference.py.
set -euo pipefail

outDir="${1:?outDir required}"
OUTPREFIX="${2:-ddim}"

nProcess="${PROCESS:?PROCESS not set}"
echo "@@ outDir=${outDir}"
echo "@@ OUTPREFIX=${OUTPREFIX}"
echo "@@ PROCESS=${nProcess}"
echo "@@ hostname=$(hostname)"
echo "@@ date=$(date -Is)"

source /cvmfs/larsoft.opensciencegrid.org/spack-packages/setup-env.sh
spack load cmake@3.27.7 || true
spack load ifdhc@2.7.2

echo "@@ ls -alh (scratch cwd)"
ls -alh

GIT_URL="${ANOMALY_GIT_URL:-https://github.com/wjdanswjddl/anomaly-detection-ana.git}"
GIT_REF="${ANOMALY_GIT_REF:-main}"

echo "@@ git clone ${GIT_URL}"
git clone "${GIT_URL}" anomaly-detection-ana
cd anomaly-detection-ana
echo "@@ git checkout ${GIT_REF}"
git checkout "${GIT_REF}"
git rev-parse --short HEAD
ls -alh

thisOutputCreationDir="$(pwd)"
filesFromSender="${CONDOR_DIR_INPUT}/bin_dir"

echo "@@ run init_grid.sh"
# shellcheck disable=SC1091
source ./bin/init_grid.sh

export IFDH_CP_MAXRETRIES="${IFDH_CP_MAXRETRIES:-2}"

echo "@@ ifdh mkdir_p ${outDir}"
ifdh mkdir_p "${outDir}" || true

echo "@@ stage worker script run_${nProcess}.sh"
cp -f "${filesFromSender}/run_${nProcess}.sh" ./
chmod +x "./run_${nProcess}.sh"

echo "@@ source run_${nProcess}.sh"
set +e
# shellcheck disable=SC1090
source "./run_${nProcess}.sh" &> "log_${nProcess}.log"
rc=$?
set -e
echo "@@ worker exit code ${rc}"
tail -n 80 "log_${nProcess}.log" || true

# Copy logs always; copy outputs if present.
ifdh cp "${thisOutputCreationDir}/log_${nProcess}.log" "${outDir}/log_${nProcess}.log" || true

if [ -d "${thisOutputCreationDir}/out_${nProcess}" ]; then
  echo "@@ shipping out_${nProcess}/"
  # Tar outputs for a single ifdh transfer (many small pickles/npz/json).
  tar czf "out_${nProcess}.tgz" -C "${thisOutputCreationDir}" "out_${nProcess}"
  ifdh cp "${thisOutputCreationDir}/out_${nProcess}.tgz" "${outDir}/out_${nProcess}.tgz"
  echo "@@ Done shipping tarball"
else
  echo "@@ ERROR: out_${nProcess} missing (inference likely failed); see log_${nProcess}.log"
  exit "${rc}"
fi

exit "${rc}"
