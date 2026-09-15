#!/bin/bash
# Watchdog for wvfm2h5 batch conversion.
# Logs progress every 10 minutes; pauses and resumes if memory pressure is detected.

set -uo pipefail

WAVEFORM_DIR="${WAVEFORM_DIR:-/pnfs/sbnd/scratch/users/munjung/v10_06_00/raw/waveforms}"
H5_DIR="${H5_DIR:-/pnfs/sbnd/scratch/users/munjung/v10_06_00/raw/h5}"
BATCH_SCRIPT="${BATCH_SCRIPT:-/exp/sbnd/app/users/munjung/anomaly-detection/samples/run_wvfm2h5_batch.sh}"
BATCH_TAG="${BATCH_TAG:-raw}"
TPC="${TPC:-0}"
LOG_DIR="/exp/sbnd/data/users/munjung/anomaly-detection/logs"
MONITOR_LOG="$LOG_DIR/wvfm2h5_monitor_${BATCH_TAG}_tpc${TPC}.log"
STATE_FILE="$LOG_DIR/wvfm2h5_monitor_state_${BATCH_TAG}_tpc${TPC}.txt"
CHECKPOINT_FILE="$LOG_DIR/wvfm2h5_checkpoint_${BATCH_TAG}_tpc${TPC}.txt"
BATCH_PID_FILE="$LOG_DIR/wvfm2h5_batch_${BATCH_TAG}_tpc${TPC}.pid"
MONITOR_PID_FILE="$LOG_DIR/wvfm2h5_monitor_${BATCH_TAG}_tpc${TPC}.pid"
WORKERS_FILE="$LOG_DIR/wvfm2h5_workers_${BATCH_TAG}_tpc${TPC}.conf"
INTERVAL_SEC=300
TARGET_USER="${USER:-munjung}"
USER_MEM_MAX_PCT=30
JOB_MEM_MAX_GB="${JOB_MEM_MAX_GB:-12}"
COOLDOWN_SEC=60

mkdir -p "$LOG_DIR"
echo $$ > "$MONITOR_PID_FILE"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MONITOR_LOG"
}

mem_available_gb() {
    awk '/^MemAvailable:/ {printf "%.1f", $2/1024/1024}' /proc/meminfo
}

mem_used_pct() {
    awk '/^MemTotal:/ {t=$2} /^MemAvailable:/ {a=$2} END {printf "%.1f", (t-a)/t*100}' /proc/meminfo
}

user_mem_rss_gb() {
    ps -u "$TARGET_USER" -o rss= 2>/dev/null | awk '{s+=$1} END {printf "%.2f", s/1024/1024}'
}

user_mem_pct() {
    local rss_kb
    rss_kb=$(ps -u "$TARGET_USER" -o rss= 2>/dev/null | awk '{s+=$1} END {print s+0}')
    awk -v rss_kb="$rss_kb" '/^MemTotal:/ {t=$2} END {if (t>0) printf "%.1f", rss_kb/t*100; else print "0.0"}' /proc/meminfo
}

job_mem_rss_gb() {
    ps -u "$TARGET_USER" -o rss=,cmd= 2>/dev/null \
        | awk '/wvfm2h5|run_wvfm2h5_batch/ {s+=$1} END {printf "%.2f", s/1024/1024}'
}

count_done() {
    local done empty
    done=$(find "$H5_DIR" -maxdepth 2 -name ".tpc${TPC}.done" 2>/dev/null | wc -l)
    empty=$(find "$H5_DIR" -maxdepth 2 -name ".tpc${TPC}.empty" 2>/dev/null | wc -l)
    echo $((done + empty))
}

count_converted() {
    find "$H5_DIR" -maxdepth 2 -name ".tpc${TPC}.done" 2>/dev/null | wc -l
}

count_total_dirs() {
    find "$WAVEFORM_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l
}

max_workers() {
    if [ -f "$WORKERS_FILE" ]; then
        cat "$WORKERS_FILE"
    else
        local budget_kb worker_kb overhead_kb workers
        budget_kb=$(awk '/^MemTotal:/ {print int($2 * 30 / 100)}' /proc/meminfo)
        worker_kb=$((10 * 1024 * 1024))
        overhead_kb=$((2 * 1024 * 1024))
        workers=$(( (budget_kb - overhead_kb) / worker_kb ))
        if [ "$workers" -lt 1 ]; then workers=1; fi
        if [ "$workers" -gt 4 ]; then workers=4; fi
        echo "$workers"
    fi
}

current_conversion() {
    tail -n 60 "$LOG_DIR/wvfm2h5_batch_${BATCH_TAG}_tpc${TPC}.log" 2>/dev/null \
        | grep -E '^(Converting |DONE |SKIP )' | tail -3 | tr '\n' '; ' | sed 's/; $//'
}

wvfm_pids() {
    pgrep -f "python3 .*wvfm2h5.py.*--save_path ${H5_DIR}/" || true
}

batch_pids() {
    if [ -f "$BATCH_PID_FILE" ]; then
        local pid
        pid=$(cat "$BATCH_PID_FILE" 2>/dev/null || true)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
        fi
    fi
}

zombie_wvfm_count() {
    ps -eo stat,cmd | awk -v h5="$H5_DIR" '/^[Zz]/ && /wvfm2h5/ && index($0, h5) {c++} END {print c+0}'
}

is_batch_running() {
    local pids
    pids=$(batch_pids)
    [ -n "$pids" ]
}

is_converter_running() {
    local pids
    pids=$(wvfm_pids)
    [ -n "$pids" ]
}

write_checkpoint() {
    local done total current avail used_pct user_pct user_gb job_gb wvfm_count zombie_count
    done=$(count_done)
    total=$(count_total_dirs)
    current=$(current_conversion)
    avail=$(mem_available_gb)
    used_pct=$(mem_used_pct)
    user_pct=$(user_mem_pct)
    user_gb=$(user_mem_rss_gb)
    job_gb=$(job_mem_rss_gb)
    wvfm_count=$(wvfm_pids | wc -w)
    zombie_count=$(zombie_wvfm_count)

    {
        echo "timestamp=$(date -Iseconds)"
        echo "done=$done"
        echo "total=$total"
        echo "current=${current:-unknown}"
        echo "mem_available_gb=$avail"
        echo "mem_used_pct=$used_pct"
        echo "user_mem_gb=$user_gb"
        echo "user_mem_pct=$user_pct"
        echo "job_mem_gb=$job_gb"
        echo "wvfm_procs=$wvfm_count"
        echo "zombie_procs=$zombie_count"
        echo "batch_running=$(is_batch_running && echo yes || echo no)"
    } > "$CHECKPOINT_FILE"

    cp "$CHECKPOINT_FILE" "$STATE_FILE"
}

log_status() {
    local done total current avail used_pct user_pct user_gb job_gb wvfm_count zombie_count workers
    done=$(count_done)
    total=$(count_total_dirs)
    current=$(current_conversion)
    avail=$(mem_available_gb)
    used_pct=$(mem_used_pct)
    user_pct=$(user_mem_pct)
    user_gb=$(user_mem_rss_gb)
    job_gb=$(job_mem_rss_gb)
    wvfm_count=$(wvfm_pids | wc -w)
    zombie_count=$(zombie_wvfm_count)
    workers=$(max_workers)

    log "PROGRESS done=$done/$total converted=$(count_converted) workers=$workers/$wvfm_count active current='${current:-idle}' user_mem=${user_gb}GiB(${user_pct}%) job_mem=${job_gb}GiB sys_used=${used_pct}% mem_avail=${avail}GiB zombies=$zombie_count batch_running=$(is_batch_running && echo yes || echo no)"
}

JOB_MEM_ONLY="${JOB_MEM_ONLY:-0}"

memory_pressure() {
    local user_pct job_gb
    user_pct=$(user_mem_pct)
    job_gb=$(job_mem_rss_gb)

    if [ "$JOB_MEM_ONLY" -eq 1 ]; then
        awk -v job="$job_gb" -v max_job="$JOB_MEM_MAX_GB" '
            BEGIN { exit (job + 0 > max_job) ? 0 : 1 }
        '
        return
    fi

    awk -v user="$user_pct" -v max_user="$USER_MEM_MAX_PCT" \
        -v job="$job_gb" -v max_job="$JOB_MEM_MAX_GB" '
        BEGIN {
            if (job + 0 > max_job) exit 0;
            if (user + 0 > max_user) exit 0;
            exit 1;
        }
    '
}

too_many_procs() {
    local wvfm_count max_workers_count
    wvfm_count=$(wvfm_pids | wc -w)
    max_workers_count=$(max_workers)
    [ "$wvfm_count" -gt "$max_workers_count" ]
}

stop_batch() {
    local reason="$1"
    log "STOPPING batch ($reason)"

    write_checkpoint

    local pid
    for pid in $(wvfm_pids); do
        log "Sending TERM to wvfm2h5.py pid=$pid"
        kill -TERM "$pid" 2>/dev/null || true
    done

    for pid in $(batch_pids); do
        log "Sending TERM to batch pid=$pid"
        kill -TERM "$pid" 2>/dev/null || true
    done

    sleep 10

    for pid in $(wvfm_pids); do
        log "Sending KILL to wvfm2h5.py pid=$pid"
        kill -KILL "$pid" 2>/dev/null || true
    done

    for pid in $(batch_pids); do
        log "Sending KILL to batch pid=$pid"
        kill -KILL "$pid" 2>/dev/null || true
    done

    sleep 2

    local zombies
    zombies=$(zombie_wvfm_count)
    if [ "$zombies" -gt 0 ]; then
        log "WARNING: $zombies zombie wvfm2h5-related processes remain; parents may need manual cleanup"
        ps -eo pid,ppid,stat,cmd | awk '/^[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+Z/ && /wvfm2h5/'
    fi

    rm -f "$BATCH_PID_FILE"
    log "Batch stopped. Checkpoint saved to $CHECKPOINT_FILE"
}

start_batch() {
    if is_batch_running || is_converter_running; then
        log "Batch already running; not starting another instance"
        return 0
    fi

    log "STARTING batch from checkpoint"
    nohup "$BATCH_SCRIPT" >> "$LOG_DIR/wvfm2h5_batch_${BATCH_TAG}_tpc${TPC}.nohup.out" 2>&1 &
    local pid=$!
    echo "$pid" > "$BATCH_PID_FILE"
    log "Started batch pid=$pid"
}

ensure_batch_running() {
    if ! is_batch_running && ! is_converter_running; then
        local done total
        done=$(count_done)
        total=$(count_total_dirs)
        if [ "$done" -lt "$total" ]; then
            start_batch
        else
            log "All directories complete ($done/$total); batch not needed"
        fi
    fi
}

log "Monitor started (pid=$$, interval=${INTERVAL_SEC}s, user=${TARGET_USER}, user_mem_max=${USER_MEM_MAX_PCT}%)"
write_checkpoint
log_status

while true; do
    sleep "$INTERVAL_SEC"

    write_checkpoint
    log_status

    done=$(count_done)
    total=$(count_total_dirs)
    if [ "$done" -ge "$total" ]; then
        log "COMPLETE: all $total directories converted"
        if is_batch_running || is_converter_running; then
            stop_batch "job complete"
        fi
        break
    fi

    if memory_pressure || too_many_procs || [ "$(zombie_wvfm_count)" -gt 0 ]; then
        reason="memory/process pressure"
        if memory_pressure; then
            reason="memory limit (user_mem=$(user_mem_pct)%, job_mem=$(job_mem_rss_gb)GiB, limits=${USER_MEM_MAX_PCT}%/${JOB_MEM_MAX_GB}GiB)"
        elif too_many_procs; then
            reason="too many wvfm2h5 processes (> $(max_workers))"
        else
            reason="zombie wvfm2h5 processes detected"
        fi

        stop_batch "$reason"
        log "Cooling down for ${COOLDOWN_SEC}s before resume"
        sleep "$COOLDOWN_SEC"
        write_checkpoint
        log_status
        start_batch
        continue
    fi

    ensure_batch_running
done

log "Monitor exiting"
rm -f "$MONITOR_PID_FILE"
