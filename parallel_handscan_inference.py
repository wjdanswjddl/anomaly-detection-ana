#!/usr/bin/env python3
"""Parallel handscan validation inference with a hard worker-memory budget.

Interleaves **healthy** and **unhealthy** jobs so both accumulate simultaneously.
Spawns as many single-file workers as fit under ``--mem-budget-gb`` (default 15),
monitors RSS, and kills/requeues stale workers.

Example::

    PYTHONPATH=_stubs:train/diffusion-anomaly \\
      python parallel_handscan_inference.py --mem-budget-gb 15
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

_REPO = Path(__file__).resolve().parent
PYTHON = Path("/exp/sbnd/app/users/munjung/env/bin/python")
WORKER_SCRIPT = _REPO / "run_handscan_validation_inference.py"


@dataclass
class Job:
    label: str
    npz: Path
    attempts: int = 0


@dataclass
class Worker:
    job: Job
    proc: subprocess.Popen
    log_path: Path
    heartbeat: Path
    started: float
    last_hb: float


def rss_kb(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
        return 0
    return 0


def tree_rss_kb(pid: int) -> int:
    """Sum RSS of pid and all descendants."""
    total = rss_kb(pid)
    try:
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return total
    for c in children:
        try:
            total += tree_rss_kb(int(c))
        except ValueError:
            continue
    return total


def pkl_path(output_root: Path, label: str, npz: Path, T: int) -> Path:
    return output_root / label / f"{npz.stem}_T{T}_ddim2ddim.pkl"


def build_queue(input_root: Path, output_root: Path, T: int) -> deque[Job]:
    """Interleave healthy / unhealthy so both sides progress together."""
    by_label: dict[str, list[Path]] = {}
    for label in ("healthy", "unhealthy"):
        done = []
        pending = []
        for nz in sorted((input_root / label).glob("*.npz")):
            if pkl_path(output_root, label, nz, T).is_file():
                done.append(nz)
            else:
                pending.append(nz)
        by_label[label] = pending
        print(f"[{label}] pending={len(pending)} already_done={len(done)}", flush=True)

    q: deque[Job] = deque()
    # Round-robin
    while any(by_label.values()):
        for label in ("healthy", "unhealthy"):
            if by_label[label]:
                q.append(Job(label=label, npz=by_label[label].pop(0)))
    return q


def kill_worker(w: Worker, reason: str) -> None:
    pid = w.proc.pid
    print(f"[kill] pid={pid} {w.job.label}/{w.job.npz.name} reason={reason}", flush=True)
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            w.proc.terminate()
        except Exception:
            pass
    deadline = time.time() + 10
    while time.time() < deadline:
        if w.proc.poll() is not None:
            break
        time.sleep(0.2)
    if w.proc.poll() is None:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                w.proc.kill()
            except Exception:
                pass
    try:
        w.heartbeat.unlink(missing_ok=True)
    except Exception:
        pass


def spawn_worker(
    job: Job,
    *,
    output_root: Path,
    model_path: Path,
    T: int,
    batch_size: int,
    max_patches: int | None,
    log_dir: Path,
    hb_dir: Path,
) -> Worker:
    log_dir.mkdir(parents=True, exist_ok=True)
    hb_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{job.label}_{job.npz.stem}_{int(time.time())}"
    log_path = log_dir / f"worker_{stamp}.log"
    heartbeat = hb_dir / f"hb_{stamp}.txt"
    heartbeat.write_text("starting\n")

    cmd = [
        str(PYTHON),
        "-u",
        str(WORKER_SCRIPT),
        "--npz",
        str(job.npz),
        "--label",
        job.label,
        "--output-root",
        str(output_root),
        "--model-path",
        str(model_path),
        "--T",
        str(T),
        "--batch-size",
        str(batch_size),
        "--heartbeat",
        str(heartbeat),
    ]
    if max_patches is not None:
        cmd.extend(["--max-patches", str(max_patches)])

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{_REPO / '_stubs'}:{_REPO / 'train' / 'diffusion-anomaly'}:{env.get('PYTHONPATH', '')}"
    # Cap torch threads so many workers do not oversubscribe
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("TORCH_NUM_THREADS", "1")

    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        cwd=str(_REPO),
        env=env,
        start_new_session=True,  # own process group for clean kill
    )
    log_fh.close()
    print(f"[spawn] pid={proc.pid} {job.label}/{job.npz.name} log={log_path.name}", flush=True)
    now = time.time()
    return Worker(job=job, proc=proc, log_path=log_path, heartbeat=heartbeat, started=now, last_hb=now)


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def kill_orphans(tracked_pids: set[int]) -> int:
    """Kill untracked run_handscan_validation_inference processes (stale leftovers)."""
    killed = 0
    try:
        out = subprocess.check_output(["ps", "-u", str(os.getuid()), "-o", "pid=,cmd="], text=True)
    except Exception:
        return 0
    for line in out.splitlines():
        if "run_handscan_validation_inference.py" not in line:
            continue
        try:
            pid = int(line.split(None, 1)[0])
        except ValueError:
            continue
        if pid in tracked_pids or pid == os.getpid():
            continue
        print(f"[orphan-kill] pid={pid}", flush=True)
        try:
            os.kill(pid, signal.SIGKILL)
            killed += 1
        except (ProcessLookupError, PermissionError):
            pass
    return killed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-root", type=Path, default=_REPO / "handscan_validation/npz_inference")
    ap.add_argument("--output-root", type=Path, default=_REPO / "handscan_validation/inference_T100")
    ap.add_argument(
        "--model-path",
        type=Path,
        default=Path("/exp/sbnd/data/users/gputnam/training-SBND/iterE/results/brats2update111000.pt"),
    )
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-patches", type=int, default=None)
    ap.add_argument("--mem-budget-gb", type=float, default=15.0, help="Max total worker RSS (GiB)")
    ap.add_argument(
        "--worker-rss-estimate-gb",
        type=float,
        default=1.3,
        help="Conservative per-worker RSS used when deciding whether to spawn",
    )
    ap.add_argument("--poll-sec", type=float, default=5.0)
    ap.add_argument(
        "--stale-sec",
        type=float,
        default=900.0,
        help="Kill worker if heartbeat older than this (seconds). One patch ≈ 3 min.",
    )
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--log-dir", type=Path, default=_REPO / "handscan_validation/logs/workers")
    ap.add_argument("--state-dir", type=Path, default=_REPO / "handscan_validation/run_state")
    args = ap.parse_args()

    budget_kb = int(args.mem_budget_gb * 1024 * 1024)
    estimate_kb = int(args.worker_rss_estimate_gb * 1024 * 1024)

    queue = build_queue(args.input_root, args.output_root, args.T)
    workers: list[Worker] = []
    finished_ok = 0
    failed: list[dict] = []

    print(
        f"Queue={len(queue)}  budget={args.mem_budget_gb:.1f} GiB  "
        f"estimate/worker={args.worker_rss_estimate_gb:.2f} GiB  "
        f"max_workers≈{max(1, budget_kb // estimate_kb)}",
        flush=True,
    )

    try:
        while queue or workers:
            now = time.time()

            # ── Reap / stale-clean ────────────────────────────────────────────
            still: list[Worker] = []
            for w in workers:
                hb_age = now - w.heartbeat.stat().st_mtime if w.heartbeat.exists() else now - w.started
                if w.heartbeat.exists():
                    w.last_hb = w.heartbeat.stat().st_mtime

                rc = w.proc.poll()
                if rc is None and hb_age > args.stale_sec:
                    kill_worker(w, f"stale heartbeat {hb_age:.0f}s")
                    w.job.attempts += 1
                    if w.job.attempts < args.max_attempts:
                        queue.append(w.job)
                        print(f"[requeue] {w.job.label}/{w.job.npz.name} attempt={w.job.attempts}", flush=True)
                    else:
                        failed.append({"label": w.job.label, "npz": str(w.job.npz), "reason": "stale"})
                    continue

                if rc is None:
                    still.append(w)
                    continue

                # exited
                out = pkl_path(args.output_root, w.job.label, w.job.npz, args.T)
                if rc == 0 and out.is_file():
                    finished_ok += 1
                    print(f"[done] {w.job.label}/{w.job.npz.name} → {out.name}  (ok={finished_ok})", flush=True)
                else:
                    w.job.attempts += 1
                    print(f"[fail] pid={w.proc.pid} rc={rc} {w.job.label}/{w.job.npz.name}", flush=True)
                    if w.job.attempts < args.max_attempts:
                        queue.append(w.job)
                    else:
                        failed.append({"label": w.job.label, "npz": str(w.job.npz), "reason": f"rc={rc}"})
                try:
                    w.heartbeat.unlink(missing_ok=True)
                except Exception:
                    pass
            workers = still

            # Sweep any untracked leftover workers (e.g. from a prior crashed orchestrator)
            kill_orphans({w.proc.pid for w in workers})

            # ── Memory accounting ─────────────────────────────────────────────
            # Charge each live worker at least the estimate so we do not over-spawn
            # while models are still loading (RSS starts tiny).
            per_rss = [max(tree_rss_kb(w.proc.pid), estimate_kb) for w in workers]
            reserved_kb = sum(per_rss)
            actual_kb = sum(tree_rss_kb(w.proc.pid) for w in workers)
            used_gb = actual_kb / (1024 * 1024)
            reserved_gb = reserved_kb / (1024 * 1024)
            headroom_kb = budget_kb - reserved_kb
            max_workers = max(1, budget_kb // estimate_kb)

            # Drop newest workers if over reserved budget or hard max
            while workers and (reserved_kb > budget_kb or len(workers) > max_workers):
                w = workers.pop()
                kill_worker(
                    w,
                    f"over budget/slots (actual={used_gb:.2f}GiB reserved={reserved_gb:.2f}GiB "
                    f"n={len(workers)+1}>{max_workers})",
                )
                w.job.attempts += 1
                if w.job.attempts < args.max_attempts:
                    queue.appendleft(w.job)
                per_rss = [max(tree_rss_kb(x.proc.pid), estimate_kb) for x in workers]
                reserved_kb = sum(per_rss)
                actual_kb = sum(tree_rss_kb(x.proc.pid) for x in workers)
                used_gb = actual_kb / (1024 * 1024)
                reserved_gb = reserved_kb / (1024 * 1024)
                headroom_kb = budget_kb - reserved_kb

            # ── Spawn while reserved headroom / slot count allows ─────────────
            while queue and headroom_kb >= estimate_kb and len(workers) < max_workers:
                job = queue.popleft()
                # Skip if another worker / prior run already finished it
                if pkl_path(args.output_root, job.label, job.npz, args.T).is_file():
                    finished_ok += 1
                    continue
                w = spawn_worker(
                    job,
                    output_root=args.output_root,
                    model_path=args.model_path,
                    T=args.T,
                    batch_size=args.batch_size,
                    max_patches=args.max_patches,
                    log_dir=args.log_dir,
                    hb_dir=args.state_dir / "heartbeats",
                )
                workers.append(w)
                headroom_kb -= estimate_kb
                reserved_kb += estimate_kb
                reserved_gb = reserved_kb / (1024 * 1024)

            write_status(
                args.state_dir / "orchestrator_status.json",
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "queue": len(queue),
                    "workers": len(workers),
                    "finished_ok": finished_ok,
                    "failed": failed,
                    "worker_rss_gb": round(used_gb, 3),
                    "reserved_rss_gb": round(reserved_gb, 3),
                    "budget_gb": args.mem_budget_gb,
                    "max_workers": int(max_workers),
                    "active": [
                        {
                            "pid": w.proc.pid,
                            "label": w.job.label,
                            "npz": w.job.npz.name,
                            "rss_gb": round(tree_rss_kb(w.proc.pid) / (1024 * 1024), 3),
                            "hb_age_s": round(now - (w.heartbeat.stat().st_mtime if w.heartbeat.exists() else w.started), 1),
                        }
                        for w in workers
                    ],
                },
            )

            n_h = len(list((args.output_root / "healthy").glob(f"*_T{args.T}_ddim2ddim.pkl")))
            n_u = len(list((args.output_root / "unhealthy").glob(f"*_T{args.T}_ddim2ddim.pkl")))
            print(
                f"[status] workers={len(workers)}/{int(max_workers)} queue={len(queue)} "
                f"rss={used_gb:.2f} reserved={reserved_gb:.2f}/{args.mem_budget_gb:.1f} GiB  "
                f"pkls healthy={n_h} unhealthy={n_u} failed={len(failed)}",
                flush=True,
            )
            time.sleep(args.poll_sec)

    except KeyboardInterrupt:
        print("Interrupted — killing workers", flush=True)
        for w in workers:
            kill_worker(w, "keyboard interrupt")

    print(f"Done. finished_ok={finished_ok} failed={len(failed)}", flush=True)
    write_status(
        args.state_dir / "orchestrator_status.json",
        {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "finished_ok": finished_ok, "failed": failed, "done": True},
    )


if __name__ == "__main__":
    main()
