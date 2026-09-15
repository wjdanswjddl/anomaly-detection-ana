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

# Condor SciToken → BEARER_TOKEN for ifdh/gfal
if [ -n "${BEARER_TOKEN_FILE:-}" ] && [ -s "${BEARER_TOKEN_FILE}" ]; then
  export BEARER_TOKEN="$(cat "${BEARER_TOKEN_FILE}")"
  echo "@@ BEARER_TOKEN_FILE=${BEARER_TOKEN_FILE} (len=${#BEARER_TOKEN})"
else
  echo "@@ WARNING: BEARER_TOKEN_FILE unset/empty"
fi

echo "@@ ls -alh (scratch cwd)"
ls -alh

filesFromSender=""
for cand in \
  "${CONDOR_DIR_INPUT:-}/bin_dir" \
  "${INPUT_TAR_DIR_LOCAL:-}" \
  "$(dirname "${INPUT_TAR_FILE:-/dev/null}")" \
  "${_CONDOR_JOB_IWD:-}/bin_dir" \
  "/srv/.unwind_0"
do
  if [ -n "${cand}" ] && [ -d "${cand}" ] && [ -f "${cand}/run_${nProcess}.sh" ]; then
    filesFromSender="${cand}"
    break
  fi
done
if [ -z "${filesFromSender}" ]; then
  echo "@@ ERROR: cannot find dropbox contents (run_${nProcess}.sh)"
  echo "@@ CONDOR_DIR_INPUT=${CONDOR_DIR_INPUT:-<unset>}"
  echo "@@ INPUT_TAR_DIR_LOCAL=${INPUT_TAR_DIR_LOCAL:-<unset>}"
  echo "@@ INPUT_TAR_FILE=${INPUT_TAR_FILE:-<unset>}"
  ls -la "${CONDOR_DIR_INPUT:-.}" 2>/dev/null || true
  ls -la "${INPUT_TAR_DIR_LOCAL:-.}" 2>/dev/null || true
  ls -la /srv/.unwind_0 2>/dev/null || true
  exit 2
fi
echo "@@ filesFromSender=${filesFromSender}"
ls -alh "${filesFromSender}"
export DROPBOX_DIR="${filesFromSender}"

# Prefer git clone when requested / when no bundle (repo is public).
# Bundle remains the fallback for offline / private-repo cases.
if [ "${ANOMALY_USE_GIT_CLONE:-1}" = "1" ] || [ ! -f "${filesFromSender}/repo_bundle.tar" ]; then
  GIT_URL="${ANOMALY_GIT_URL:-https://github.com/wjdanswjddl/anomaly-detection-ana.git}"
  GIT_REF="${ANOMALY_GIT_REF:-main}"
  echo "@@ git clone ${GIT_URL}"
  git clone --depth 1 --branch "${GIT_REF}" "${GIT_URL}" anomaly-detection-ana \
    || { git clone "${GIT_URL}" anomaly-detection-ana && git -C anomaly-detection-ana checkout "${GIT_REF}"; }
  cd anomaly-detection-ana
  git rev-parse --short HEAD
else
  echo "@@ extracting ${filesFromSender}/repo_bundle.tar"
  mkdir -p anomaly-detection-ana
  tar xf "${filesFromSender}/repo_bundle.tar" -C anomaly-detection-ana
  cd anomaly-detection-ana
fi
ls -alh

thisOutputCreationDir="$(pwd)"
logFile="${thisOutputCreationDir}/log_${nProcess}.log"

upload_log() {
  if [ -f "${logFile}" ]; then
    echo "@@ ifdh cp log → ${outDir}/log_${nProcess}.log"
    ifdh cp "${logFile}" "${outDir}/log_${nProcess}.log" || true
  fi
}
trap upload_log EXIT

echo "@@ run init_grid.sh"
# shellcheck disable=SC1091
source ./bin/init_grid.sh

export IFDH_CP_MAXRETRIES="${IFDH_CP_MAXRETRIES:-2}"

echo "@@ ifdh mkdir_p ${outDir}"
ifdh mkdir_p "${outDir}" || true

echo "@@ stage worker script + model from dropbox"
cp -f "${filesFromSender}/run_${nProcess}.sh" ./
chmod +x "./run_${nProcess}.sh"
if [ -f "${filesFromSender}/model.pt" ]; then
  cp -f "${filesFromSender}/model.pt" ./model.pt
  ls -lh ./model.pt
else
  echo "@@ WARNING: model.pt not in dropbox; worker will ifdh it"
fi

# IMPORTANT: run as a subprocess (not `source`). The worker script uses
# `set -e`; sourcing it would abort this parent before logs are uploaded.
echo "@@ bash run_${nProcess}.sh"
set +e
bash "./run_${nProcess}.sh" > "${logFile}" 2>&1
rc=$?
set -e
echo "@@ worker exit code ${rc}"
tail -n 120 "${logFile}" || true

if [ -d "${thisOutputCreationDir}/out_${nProcess}" ] && \
   find "${thisOutputCreationDir}/out_${nProcess}" -type f | grep -q .; then
  echo "@@ shipping out_${nProcess}/"
  tar czf "out_${nProcess}.tgz" -C "${thisOutputCreationDir}" "out_${nProcess}"
  ifdh cp "${thisOutputCreationDir}/out_${nProcess}.tgz" "${outDir}/out_${nProcess}.tgz"
  echo "@@ Done shipping tarball"
else
  echo "@@ ERROR: out_${nProcess} empty or missing (inference likely failed); see log_${nProcess}.log"
  exit "${rc:-1}"
fi

exit "${rc}"
