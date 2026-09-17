#!/usr/bin/env bash
# Stop diffusion / AE trainers on the EAF Jupyter host.
# Run this IN a terminal on jupyter-munjung (EAF), not on sbndbuild*.
set -euo pipefail

APP_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STOP="$APP_ROOT/STOP_TRAINING"

echo "hostname: $(hostname)"
echo "before:"
ps -u "$USER" -o pid,etime,cmd | grep -E 'image_train|autoencoder_train' | grep -v grep || echo "  (no matching trainers)"

pkill -f 'scripts/image_train.py' 2>/dev/null || true
pkill -f 'scripts/autoencoder_train.py' 2>/dev/null || true
sleep 2
pkill -9 -f 'scripts/image_train.py' 2>/dev/null || true
pkill -9 -f 'scripts/autoencoder_train.py' 2>/dev/null || true

touch "$STOP"
echo "wrote cooperative stop file: $STOP"

echo "after:"
ps -u "$USER" -o pid,etime,cmd | grep -E 'image_train|autoencoder_train' | grep -v grep || echo "  (no matching trainers)"
echo "Also interrupt/shutdown the 01a and 01b Jupyter kernels if their cells are still blocked."
echo "Before starting a new full run: rm -f $STOP"
