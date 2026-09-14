#!/bin/bash
# Rebuild dump_waveform.tar with wvfm2h5 C++ converter included.
set -euo pipefail

SRC_DIR="/exp/sbnd/app/users/munjung/anomaly-detection/grid_wvfm2h5"
ING_DIR="/exp/sbnd/app/users/munjung/v10_06_00_02-dnnsp/jobs/tarball_ingredients"
OUT_TAR="/pnfs/sbnd/resilient/users/munjung/dump_waveform.tar"
STAGE=$(mktemp -d)

cp -a "$ING_DIR/dump_waveform_all.C" "$STAGE/"
cp -a "$ING_DIR/setup.sh" "$STAGE/"
cp -a "$ING_DIR/wirecell_init.sh" "$STAGE/"
cp -a "$ING_DIR/rename_h5s.sh" "$STAGE/" 2>/dev/null || true

# New / updated pieces
cp -a "$SRC_DIR/wvfm2h5.cpp" "$STAGE/"
cp -a "$SRC_DIR/wvfm2h5_run.sh" "$STAGE/"
cp -a "$SRC_DIR/wirecell_macro.sh" "$STAGE/"
chmod +x "$STAGE/wvfm2h5_run.sh" "$STAGE/wirecell_macro.sh" "$STAGE/wirecell_init.sh"

# Keep production endscript in sync
cp -a "$SRC_DIR/wirecell_macro.sh" "$ING_DIR/wirecell_macro.sh"
cp -a "$SRC_DIR/wvfm2h5.cpp" "$ING_DIR/wvfm2h5.cpp"
cp -a "$SRC_DIR/wvfm2h5_run.sh" "$ING_DIR/wvfm2h5_run.sh"
chmod +x "$ING_DIR/wvfm2h5_run.sh" "$ING_DIR/wirecell_macro.sh"

tar -cf "$OUT_TAR" -C "$STAGE" .
rm -rf "$STAGE"

echo "Wrote $OUT_TAR"
tar -tf "$OUT_TAR"
