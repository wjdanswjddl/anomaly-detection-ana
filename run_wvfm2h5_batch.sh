#!/bin/bash
set -uo pipefail

BATCH_TAG="${BATCH_TAG:-raw}"
WAVEFORM_DIR="${WAVEFORM_DIR:-/pnfs/sbnd/scratch/users/munjung/v10_06_00/raw/waveforms}"
H5_DIR="${H5_DIR:-/pnfs/sbnd/scratch/users/munjung/v10_06_00/raw/h5}"
SCRIPT="/exp/sbnd/app/users/munjung/anomaly-detection/wvfm2h5.py"
TPC="${TPC:-0}"
N_EVENTS_PER_FILE=10
MIN_WAVEFORM_BYTES=32768
LOG_DIR="/exp/sbnd/app/users/munjung/anomaly-detection/logs"
LOG_FILE="$LOG_DIR/wvfm2h5_batch_${BATCH_TAG}_tpc${TPC}.log"
BATCH_PID_FILE="$LOG_DIR/wvfm2h5_batch_${BATCH_TAG}_tpc${TPC}.pid"
LOCK_FILE="$LOG_DIR/wvfm2h5_batch_${BATCH_TAG}_tpc${TPC}.lock"
WORKERS_FILE="$LOG_DIR/wvfm2h5_workers_${BATCH_TAG}_tpc${TPC}.conf"
LOG_LOCK_FILE="$LOG_DIR/wvfm2h5_batch_${BATCH_TAG}_tpc${TPC}.log.lock"

USE_PARALLEL="${USE_PARALLEL:-0}"
PAUSE_EVERY="${PAUSE_EVERY:-3}"
PAUSE_COOLDOWN_SEC="${PAUSE_COOLDOWN_SEC:-45}"

# Sequential mode: 1 worker. Parallel only if explicitly enabled.
MEM_TOTAL_KB=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
USER_MEM_BUDGET_KB=$((MEM_TOTAL_KB * 30 / 100))
OVERHEAD_KB=$((2 * 1024 * 1024))
WORKER_MEM_KB=$((12 * 1024 * 1024))
CURRENT_USER_KB=$(ps -u "${USER:-munjung}" -o rss= 2>/dev/null | awk '{s+=$1} END {print s+0}')
REMAINING_KB=$((USER_MEM_BUDGET_KB - CURRENT_USER_KB))
DEFAULT_WORKERS=$(( (REMAINING_KB - OVERHEAD_KB) / WORKER_MEM_KB ))
if [ "$DEFAULT_WORKERS" -lt 1 ]; then DEFAULT_WORKERS=1; fi
if [ "$DEFAULT_WORKERS" -gt 3 ]; then DEFAULT_WORKERS=3; fi
if [ "$USE_PARALLEL" -eq 0 ]; then
    DEFAULT_WORKERS=1
fi
N_WORKERS="${N_WORKERS:-$DEFAULT_WORKERS}"

mkdir -p "$LOG_DIR" "$H5_DIR"
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    echo "Another ${BATCH_TAG} tpc${TPC} batch is already running; exiting."
    exit 0
fi

echo $$ > "$BATCH_PID_FILE"
echo "$N_WORKERS" > "$WORKERS_FILE"
trap 'rm -f "$BATCH_PID_FILE"' EXIT

log_msg() {
    { flock 9; echo "$*"; } 9>>"$LOG_LOCK_FILE" >>"$LOG_FILE"
}

cleanup_job_procs() {
    local pid zombies
    for pid in $(pgrep -f "python3 .*wvfm2h5.py.*--save_path ${H5_DIR}/" 2>/dev/null || true); do
        kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in $(pgrep -f "python3 .*wvfm2h5.py.*--save_path ${H5_DIR}/" 2>/dev/null || true); do
        kill -KILL "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    zombies=$(ps -eo stat,cmd | awk -v h5="$H5_DIR" '/^[Zz]/ && /wvfm2h5/ && index($0, h5) {c++} END {print c+0}')
    if [ "$zombies" -gt 0 ]; then
        log_msg "WARNING: $zombies zombie wvfm2h5 processes after cleanup"
    fi
}

pause_and_cleanup() {
    local n="$1"
    log_msg "PAUSE after $n conversions: cooldown ${PAUSE_COOLDOWN_SEC}s + cleanup"
    cleanup_job_procs
    sleep "$PAUSE_COOLDOWN_SEC"
    gc_collect_hint() {
        python3 -c "import gc; gc.collect()" 2>/dev/null || true
    }
    gc_collect_hint
}

waveform_is_nonempty() {
    local wf="$1/waveform.root"
    local size

    if [ ! -f "$wf" ]; then
        return 1
    fi

    size=$(stat -c%s "$wf" 2>/dev/null || echo 0)
    [ "$size" -ge "$MIN_WAVEFORM_BYTES" ]
}

convert_one() {
    local subdir="$1"
    local name outdir done_marker empty_marker
    name=$(basename "$subdir")
    outdir="$H5_DIR/$name"
    done_marker="$outdir/.tpc${TPC}.done"
    empty_marker="$outdir/.tpc${TPC}.empty"

    if [ -f "$done_marker" ] || [ -f "$empty_marker" ]; then
        return 2
    fi

    if [ ! -f "$subdir/waveform.root" ]; then
        mkdir -p "$outdir"
        touch "$empty_marker"
        log_msg "SKIP (no waveform.root): $name"
        return 2
    fi

    if ! waveform_is_nonempty "$subdir"; then
        mkdir -p "$outdir"
        touch "$empty_marker"
        log_msg "SKIP (empty waveform.root): $name"
        return 2
    fi

    mkdir -p "$outdir"
    rm -f "$outdir"/g4-raw-${TPC}_*.h5

    log_msg "Converting $name at $(date)"
    if python3 "$SCRIPT" \
        --this_dir "$subdir" \
        --save_path "$outdir" \
        --tpc "$TPC" \
        --n_events_per_file "$N_EVENTS_PER_FILE"; then
        touch "$done_marker"
        log_msg "DONE $name at $(date)"
        return 0
    fi

    log_msg "FAILED $name at $(date)"
    return 1
}

export WAVEFORM_DIR H5_DIR SCRIPT TPC N_EVENTS_PER_FILE LOG_FILE LOG_LOCK_FILE MIN_WAVEFORM_BYTES
export -f convert_one log_msg waveform_is_nonempty

log_msg "Starting batch (${BATCH_TAG}, tpc=${TPC}) at $(date) parallel=$USE_PARALLEL workers=$N_WORKERS pause_every=$PAUSE_EVERY"
log_msg "Input:  $WAVEFORM_DIR"
log_msg "Output: $H5_DIR (g4-raw-${TPC}_*.h5)"
log_msg "User memory budget: $((USER_MEM_BUDGET_KB / 1024 / 1024)) GiB, currently used: $((CURRENT_USER_KB / 1024 / 1024)) GiB"

total=0
skipped=0
empty_skipped=0
converted=0
failed=0
running=0
since_pause=0

for subdir in "$WAVEFORM_DIR"/*; do
    [ -d "$subdir" ] || continue
    name=$(basename "$subdir")
    done_marker="$H5_DIR/$name/.tpc${TPC}.done"
    empty_marker="$H5_DIR/$name/.tpc${TPC}.empty"
    total=$((total + 1))

    if [ -f "$done_marker" ] || [ -f "$empty_marker" ]; then
        skipped=$((skipped + 1))
        continue
    fi

    if [ -f "$subdir/waveform.root" ] && ! waveform_is_nonempty "$subdir"; then
        mkdir -p "$H5_DIR/$name"
        touch "$empty_marker"
        empty_skipped=$((empty_skipped + 1))
        log_msg "SKIP (empty waveform.root): $name"
        continue
    fi

    if [ "$USE_PARALLEL" -eq 1 ] && [ "$N_WORKERS" -gt 1 ]; then
        while [ "$running" -ge "$N_WORKERS" ]; do
            if wait -n 2>/dev/null; then
                converted=$((converted + 1))
                since_pause=$((since_pause + 1))
            else
                failed=$((failed + 1))
            fi
            running=$((running - 1))
            if [ "$since_pause" -ge "$PAUSE_EVERY" ]; then
                pause_and_cleanup "$since_pause"
                since_pause=0
            fi
        done
        convert_one "$subdir" &
        running=$((running + 1))
    else
        convert_one "$subdir"
        rc=$?
        if [ "$rc" -eq 0 ]; then
            converted=$((converted + 1))
            since_pause=$((since_pause + 1))
        elif [ "$rc" -eq 2 ]; then
            skipped=$((skipped + 1))
        else
            failed=$((failed + 1))
            since_pause=$((since_pause + 1))
        fi
        if [ "$since_pause" -ge "$PAUSE_EVERY" ]; then
            pause_and_cleanup "$since_pause"
            since_pause=0
        fi
    fi
done

if [ "$USE_PARALLEL" -eq 1 ] && [ "$running" -gt 0 ]; then
    while [ "$running" -gt 0 ]; do
        if wait -n 2>/dev/null; then
            converted=$((converted + 1))
        else
            failed=$((failed + 1))
        fi
        running=$((running - 1))
    done
fi

cleanup_job_procs
log_msg "Finished at $(date)"
log_msg "total=$total skipped=$skipped empty_skipped=$empty_skipped converted=$converted failed=$failed workers=$N_WORKERS"
