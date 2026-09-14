#!/bin/bash
# Build and run wvfm2h5.exe on a larsoft worker after waveform.root exists.
set -euo pipefail

INPUT="${1:-waveform.root}"
OUTDIR="${2:-.}"
N_PER_FILE="${3:-10}"

# Source lives in the dropbox tarball (often on read-only CVMFS).
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${INPUT_TAR_DIR_LOCAL_1:-}" ] && [ -f "${INPUT_TAR_DIR_LOCAL_1}/wvfm2h5.cpp" ]; then
  SRC_DIR="${INPUT_TAR_DIR_LOCAL_1}"
fi
SRC="${SRC_DIR}/wvfm2h5.cpp"

if [ ! -f "$INPUT" ]; then
  echo "wvfm2h5_run: no $INPUT — skipping"
  exit 0
fi
if [ ! -f "$SRC" ]; then
  echo "wvfm2h5_run: ERROR missing source $SRC"
  exit 1
fi

# Prefer already-setup HDF5 from the larsoft worker env; else UPS; else hard path.
if [ -z "${HDF5_INC:-}" ] || [ -z "${HDF5_LIB:-}" ]; then
  if type setup >/dev/null 2>&1; then
    setup hdf5 v1_12_2a -q e26:prof >/dev/null 2>&1 \
      || setup hdf5 v1_12_2b -q e26:prof >/dev/null 2>&1 \
      || true
  fi
fi
if [ -n "${HDF5_INC:-}" ] && [ -n "${HDF5_LIB:-}" ]; then
  HDF5_ROOT="$(dirname "$HDF5_INC")"
elif [ -d /cvmfs/larsoft.opensciencegrid.org/products/hdf5/v1_12_2a/Linux64bit+3.10-2.17-e26-prof ]; then
  HDF5_ROOT=/cvmfs/larsoft.opensciencegrid.org/products/hdf5/v1_12_2a/Linux64bit+3.10-2.17-e26-prof
  export HDF5_INC="$HDF5_ROOT/include"
  export HDF5_LIB="$HDF5_ROOT/lib"
elif [ -d /cvmfs/larsoft.opensciencegrid.org/products/hdf5/v1_12_2b/Linux64bit+3.10-2.17-e26-prof ]; then
  HDF5_ROOT=/cvmfs/larsoft.opensciencegrid.org/products/hdf5/v1_12_2b/Linux64bit+3.10-2.17-e26-prof
  export HDF5_INC="$HDF5_ROOT/include"
  export HDF5_LIB="$HDF5_ROOT/lib"
else
  HDF5_ROOT=""
fi

if [ -z "$HDF5_ROOT" ] || ! command -v root-config >/dev/null 2>&1; then
  echo "wvfm2h5_run: ERROR need root-config and hdf5 (HDF5_ROOT='$HDF5_ROOT')"
  exit 1
fi

# Always compile into a writable directory (dropbox/CVMFS is read-only).
BUILD_DIR="${PWD:-.}"
EXE="${BUILD_DIR}/wvfm2h5.exe"

echo "wvfm2h5_run: SRC=$SRC"
echo "wvfm2h5_run: EXE=$EXE"
echo "wvfm2h5_run: HDF5_INC=$HDF5_INC"
echo "wvfm2h5_run: compiling into writable dir $BUILD_DIR"
g++ -O2 -std=c++17 -o "$EXE" "$SRC" \
  $(root-config --cflags --libs) \
  -I"$HDF5_INC" -L"$HDF5_LIB" -lhdf5 \
  -Wl,-rpath,"$HDF5_LIB"
echo "wvfm2h5_run: compile OK"

echo "wvfm2h5_run: converting $INPUT -> $OUTDIR (n_events_per_file=$N_PER_FILE)"
"$EXE" --input "$INPUT" --outdir "$OUTDIR" --n_events_per_file "$N_PER_FILE"
ls -lh "$OUTDIR"/g4-raw-*.h5 2>/dev/null || echo "wvfm2h5_run: no h5 outputs (empty input?)"
