# Grid CPU inference (jobsub)

Yes — **inputs do not have to be ROOT**. `jobsub` / `ifdh` only move bytes.
This workflow stages `.npz` files (and a `.pt` checkpoint) onto the worker with
`ifdh cp`, then runs `inference/run_ddim2ddim_inference.py`.

## Layout (pnfs)

```text
/pnfs/sbnd/scratch/users/$USER/anomaly-detection/
  models/emabrats2update_0.9999_111000.pt
  lists/my_inputs.list          # one pnfs/xrootd NPZ path per line
  out/                          # ANOMALY_GRID_OUT_DIR
```

Stage the Gray EMA model once (from a machine with `/exp` + `/pnfs`):

```bash
mkdir -p /pnfs/sbnd/scratch/users/$USER/anomaly-detection/{models,lists,out}
cp -n /exp/sbnd/data/users/gputnam/training-SBND/iterE/results/emabrats2update_0.9999_111000.pt \
  /pnfs/sbnd/scratch/users/$USER/anomaly-detection/models/
```

## Submit

Requires: `jobsub` token setup (same as cafpyana), and a **cloneable** GitHub repo
(`ANOMALY_GIT_URL`). If the repo is private, make it public or arrange worker auth.

```bash
cd /exp/sbnd/app/users/munjung/anomaly-detection
export ANOMALY_WD=$PWD
export ANOMALY_GRID_OUT_DIR=/pnfs/sbnd/scratch/users/$USER/anomaly-detection/out
export ANOMALY_GIT_URL=https://github.com/wjdanswjddl/anomaly-detection-ana.git
export ANOMALY_GIT_REF=main

# Optional resource overrides (CPU T=200 is slow — prefer long lifetime)
export JOBSUB_MEMORY=12GB
export JOBSUB_DISK=20GB
export JOBSUB_LIFETIME=12h
export JOBSUB_CPU=4

python inference/submit_ddim_grid.py \
  -l /pnfs/sbnd/scratch/users/$USER/anomaly-detection/lists/handscan_test.list \
  --model /pnfs/sbnd/scratch/users/$USER/anomaly-detection/models/emabrats2update_0.9999_111000.pt \
  -o handscan_T200_test \
  -ngrid 2 \
  --T 200 --batch-size 1 \
  --dry-run   # remove after inspecting MasterJobDir
```

Monitor: `jobsub_q -G sbnd --user $USER`  
Outputs: `$ANOMALY_GRID_OUT_DIR/inference/<stamp>__<name>/out_*.tgz` (+ `log_*.log`).

## Worker flow

`bin/grid_executable.sh` → `git clone` → `bin/init_grid.sh` (venv + CPU torch) →
`run_${PROCESS}.sh` (`ifdh` NPZs + model → `run_ddim2ddim_inference.py`) → `ifdh` tarball back.
