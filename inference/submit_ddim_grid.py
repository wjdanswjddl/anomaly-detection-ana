#!/usr/bin/env python3
"""Submit SBND jobsub campaigns for CPU DDIM→DDIM inference on *.npz inputs.

Mirrors the cafpyana ``run_df_maker.py -ngrid`` pattern:

* Split a file list across ``-ngrid`` workers
* Bundle per-job ``run_N.sh`` + ``grid_executable.sh`` in a dropbox tarball
* Workers ``git clone`` this repo, ``ifdh cp`` NPZ inputs + model, run inference, ``ifdh`` results

Inputs do **not** need to be ROOT — any files ``ifdh`` can copy work (here: ``.npz``).

Example (after staging model + file list on pnfs)::

  export ANOMALY_WD=/exp/sbnd/app/users/munjung/anomaly-detection
  export ANOMALY_GRID_OUT_DIR=/pnfs/sbnd/scratch/users/munjung/anomaly-detection/out
  export ANOMALY_GIT_URL=https://github.com/wjdanswjddl/anomaly-detection-ana.git
  export ANOMALY_GIT_REF=main

  python inference/submit_ddim_grid.py \\
    -l /pnfs/sbnd/scratch/users/munjung/anomaly-detection/lists/handscan_test.list \\
    --model /pnfs/sbnd/scratch/users/munjung/anomaly-detection/models/emabrats2update_0.9999_111000.pt \\
    -o handscan_T200_test \\
    -ngrid 2 \\
    --T 200 --batch-size 1

Dry-run (write scripts/tarball, do not jobsub)::

  python inference/submit_ddim_grid.py ... -ngrid 2 --dry-run
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def _read_list(path: Path) -> list[str]:
    lines: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    if not lines:
        sys.exit(f"No input paths in {path}")
    return lines


def _chunk(items: list[str], ngrid: int) -> list[list[str]]:
    ngrid = max(1, min(ngrid, len(items)))
    buckets: list[list[str]] = [[] for _ in range(ngrid)]
    for i, item in enumerate(items):
        buckets[i % ngrid].append(item)
    return buckets


def _write_worker_script(
    path: Path,
    *,
    job_idx: int,
    files: list[str],
    model_pnfs: str,
    T: int,
    batch_size: int,
    formats: str,
    extra_args: str,
) -> None:
    """Bash executed (not sourced) on the worker after the repo + venv are ready."""
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"echo '[run_{job_idx}.sh] starting'",
        "mkdir -p inputs",
        f"mkdir -p out_{job_idx}",
        "",
        "# Model: prefer dropbox copy staged by grid_executable; else ifdh from pnfs",
        "if [ -f ./model.pt ]; then",
        f"  echo '[run_{job_idx}.sh] using dropbox model.pt'",
        "else",
        f"  echo '[run_{job_idx}.sh] ifdh cp model'",
        f"  ifdh cp {shlex.quote(model_pnfs)} ./model.pt",
        "fi",
        "ls -lh ./model.pt",
        "",
        "# NPZ inputs: dropbox (${DROPBOX_DIR}/inputs/) if present, else ifdh",
    ]
    for i, f in enumerate(files):
        base = Path(f).name
        lines += [
            f"echo '[run_{job_idx}.sh] input {i}: {f}'",
            f"if [ -n \"${{DROPBOX_DIR:-}}\" ] && [ -f \"${{DROPBOX_DIR}}/inputs/{base}\" ]; then",
            f"  cp -f \"${{DROPBOX_DIR}}/inputs/{base}\" ./inputs/{base}",
            "else",
            f"  ifdh cp {shlex.quote(f)} ./inputs/{shlex.quote(base)}",
            "fi",
        ]
    lines += [
        "ls -lh inputs/",
        "",
        "export PYTHONPATH=\"${ANOMALY_WD}/_stubs:${ANOMALY_WD}/train/diffusion-anomaly:${ANOMALY_WD}/inference:${ANOMALY_WD}:${PYTHONPATH:-}\"",
        "export OMP_NUM_THREADS=\"${OMP_NUM_THREADS:-4}\"",
        "export OPENBLAS_NUM_THREADS=\"${OPENBLAS_NUM_THREADS:-4}\"",
        "",
        "python inference/run_ddim2ddim_inference.py \\",
        "  --input-dir ./inputs \\",
        f"  --output-dir ./out_{job_idx} \\",
        "  --model-path ./model.pt \\",
        f"  --T {int(T)} \\",
        f"  --batch-size {int(batch_size)} \\",
        f"  --formats {shlex.quote(formats)}",
    ]
    if extra_args.strip():
        # Attach extra args on a continuation line.
        lines[-1] += " \\"
        lines.append(f"  {extra_args.strip()}")
    lines += [
        "",
        f"echo '[run_{job_idx}.sh] outputs:'",
        f"find out_{job_idx} -type f | head -50",
        f"echo '[run_{job_idx}.sh] done'",
    ]
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o755)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-l", "--file-list", type=Path, required=True, help="Text file: one NPZ path per line (pnfs or xrootd)")
    p.add_argument("--model", type=Path, required=True, help="Model checkpoint on pnfs (ifdh-readable)")
    p.add_argument("-o", "--output", required=True, help="Campaign / output name prefix")
    p.add_argument("-ngrid", dest="ngrid", type=int, required=True, help="Number of grid jobs (>0)")
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--formats", default="pickle,npz")
    p.add_argument(
        "--extra-args",
        default="",
        help="Extra CLI args appended to run_ddim2ddim_inference.py (quoted string)",
    )
    p.add_argument("--dry-run", action="store_true", help="Build tarball/scripts but do not jobsub_submit")
    p.add_argument("-N", "--max-files", type=int, default=0, help="Optional cap on input list length")
    args = p.parse_args()

    if args.ngrid <= 0:
        sys.exit("-ngrid must be > 0")

    wd = Path(os.environ.get("ANOMALY_WD", Path(__file__).resolve().parents[1])).resolve()
    grid_out = Path(
        os.environ.get(
            "ANOMALY_GRID_OUT_DIR",
            f"/pnfs/sbnd/scratch/users/{os.environ.get('USER', 'munjung')}/anomaly-detection/out",
        )
    )
    git_url = os.environ.get(
        "ANOMALY_GIT_URL", "https://github.com/wjdanswjddl/anomaly-detection-ana.git"
    )
    git_ref = os.environ.get("ANOMALY_GIT_REF", "main")

    exe = wd / "bin" / "grid_executable.sh"
    init = wd / "bin" / "init_grid.sh"
    if not exe.is_file() or not init.is_file():
        sys.exit(f"Missing {exe} or {init}")

    inputs = _read_list(args.file_list)
    if args.max_files and args.max_files > 0:
        inputs = inputs[: args.max_files]

    model = str(args.model)
    if not model.startswith("/pnfs/") and not model.startswith("xroot:"):
        print(
            f"WARNING: model path {model!r} is not under /pnfs or xroot — "
            "grid workers often cannot read /exp. Prefer pnfs.",
            file=sys.stderr,
        )

    buckets = _chunk(inputs, args.ngrid)
    ngrid = len(buckets)
    stamp = dt.datetime.now().strftime("%Y_%m_%d_%H%M%S")
    master = grid_out / "logs" / f"{stamp}__{args.output}_log"
    out_dir = grid_out / "inference" / f"{stamp}__{args.output}"
    master.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"ANOMALY_WD={wd}")
    print(f"ANOMALY_GRID_OUT_DIR={grid_out}")
    print(f"ANOMALY_GIT_URL={git_url}")
    print(f"ANOMALY_GIT_REF={git_ref}")
    print(f"inputs={len(inputs)} ngrid={ngrid}")
    print(f"MasterJobDir={master}")
    print(f"OutputDir={out_dir}")

    for i, flist in enumerate(buckets):
        _write_worker_script(
            master / f"run_{i}.sh",
            job_idx=i,
            files=flist,
            model_pnfs=model,
            T=args.T,
            batch_size=args.batch_size,
            formats=args.formats,
            extra_args=args.extra_args,
        )
        print(f"  job {i}: {len(flist)} file(s)")

    shutil.copy2(exe, master / "grid_executable.sh")
    (master / "grid_executable.sh").chmod(0o755)

    # Stage model into dropbox (avoids a large ifdh pull on every worker).
    model_drop = master / "model.pt"
    print(f"Copying model → {model_drop}")
    shutil.copy2(model, model_drop)

    # Stage NPZ inputs into dropbox when small enough (smoke / modest lists).
    # Larger campaigns still fall back to ifdh inside run_*.sh.
    inputs_dir = master / "inputs"
    inputs_dir.mkdir(exist_ok=True)
    stage_bytes = 0
    staged_inputs = 0
    max_stage = int(os.environ.get("ANOMALY_MAX_STAGE_BYTES", str(400 * 1024 * 1024)))
    for src in inputs:
        p = Path(src)
        try:
            sz = p.stat().st_size
        except OSError:
            print(f"WARNING: cannot stat {src}; worker will ifdh it", file=sys.stderr)
            continue
        if stage_bytes + sz > max_stage:
            print(f"Skipping further input staging (>{max_stage/1e6:.0f} MB); remaining via ifdh")
            break
        dest = inputs_dir / p.name
        if not dest.exists():
            shutil.copy2(p, dest)
        stage_bytes += sz
        staged_inputs += 1
    print(f"Staged {staged_inputs}/{len(inputs)} inputs ({stage_bytes/1e6:.1f} MB) into dropbox")

    # Optional slim repo snapshot (fallback if git clone is disabled/unavailable).
    repo_bundle = master / "repo_bundle.tar"
    if repo_bundle.exists():
        repo_bundle.unlink()
    bundle_members = [
        "bin",
        "inference",
        "_stubs",
        "configs",
        "train/diffusion-anomaly/guided_diffusion",
        "train/diffusion-anomaly/scripts",
        "train/diffusion-anomaly/requirements.txt",
        "train/diffusion-anomaly/README.md",
        "train/diffusion-anomaly/LICENSE",
    ]
    for extra in sorted((wd / "train" / "diffusion-anomaly").glob("*.sh")):
        bundle_members.append(f"train/diffusion-anomaly/{extra.name}")
    missing = [m for m in bundle_members if not (wd / m).exists()]
    if missing:
        sys.exit(f"Cannot build repo_bundle.tar; missing: {missing}")
    print(f"Building {repo_bundle} ({len(bundle_members)} members)…")
    subprocess.check_call(
        ["tar", "cf", str(repo_bundle), *bundle_members],
        cwd=str(wd),
    )
    print(f"  repo_bundle.tar size={repo_bundle.stat().st_size / 1e6:.1f} MB")

    # Record campaign metadata for humans.
    (master / "campaign.txt").write_text(
        "\n".join(
            [
                f"stamp={stamp}",
                f"output={args.output}",
                f"model={model}",
                f"T={args.T}",
                f"batch_size={args.batch_size}",
                f"git_url={git_url}",
                f"git_ref={git_ref}",
                f"n_inputs={len(inputs)}",
                f"ngrid={ngrid}",
                f"file_list={args.file_list}",
                f"repo_bundle={repo_bundle}",
                f"staged_inputs={staged_inputs}",
                f"model_in_dropbox={model_drop}",
                "",
            ]
        )
    )

    cwd = Path.cwd()
    try:
        os.chdir(master)
        tar_path = master / "bin_dir.tar"
        if tar_path.exists():
            tar_path.unlink()
        members = [
            "grid_executable.sh",
            "campaign.txt",
            "repo_bundle.tar",
            "model.pt",
        ] + [f"run_{i}.sh" for i in range(ngrid)]
        if staged_inputs:
            members.append("inputs")
        subprocess.check_call(["tar", "cf", "bin_dir.tar", *members])
        print(f"  bin_dir.tar size={(master / 'bin_dir.tar').stat().st_size / 1e6:.1f} MB")
    finally:
        os.chdir(cwd)

    job_disk = os.environ.get("JOBSUB_DISK", "20GB")
    job_mem = os.environ.get("JOBSUB_MEMORY", "6GB")
    job_life = os.environ.get("JOBSUB_LIFETIME", "12h")
    job_cpu = os.environ.get("JOBSUB_CPU", "4")
    use_git = os.environ.get("ANOMALY_USE_GIT_CLONE", "1")

    # Pass git URL/ref into the worker environment.
    submit_cmd = f"""jobsub_submit \\
-G sbnd \\
--auth-methods="token" \\
-e LC_ALL=C \\
-e ANOMALY_GIT_URL={shlex.quote(git_url)} \\
-e ANOMALY_GIT_REF={shlex.quote(git_ref)} \\
-e ANOMALY_USE_GIT_CLONE={shlex.quote(use_git)} \\
--role=Analysis \\
--resource-provides="usage_model=DEDICATED,OPPORTUNISTIC" \\
--lines '+FERMIHTC_AutoRelease=True' --lines '+FERMIHTC_GraceMemory=5000' --lines '+FERMIHTC_GraceLifetime=3600' \\
--append_condor_requirements='(TARGET.HAS_SINGULARITY=?=true)' \\
--use-pnfs-dropbox \\
--tar_file_name "dropbox://{master}/bin_dir.tar" \\
-N {ngrid} \\
--disk {job_disk} \\
--cpu {job_cpu} \\
--memory {job_mem} \\
--expected-lifetime {job_life} \\
"file://{master}/grid_executable.sh" \\
"{out_dir}" \\
"{args.output}"
"""
    print(submit_cmd)
    if args.dry_run:
        print("DRY-RUN: not submitting")
        return

    # jobsub must run with the master dir as CWD for relative dropbox paths in some setups;
    # we used absolute paths above.
    subprocess.check_call(submit_cmd, shell=True)
    print(f"Submitted. Monitor: jobsub_q -G sbnd --user {os.environ.get('USER', '$USER')}")
    print(f"Outputs (when done): {out_dir}/out_*.tgz")

if __name__ == "__main__":
    main()
