#!/usr/bin/env python3
"""Poll for synced training EMA ckpts on pnfs, then submit grid jobs."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/pnfs/sbnd/scratch/users/munjung/anomaly-detection/models_training")
REPO = Path("/exp/sbnd/app/users/munjung/anomaly-detection")
FLAG = Path(
    "/exp/sbnd/data/users/munjung/anomaly-detection/training/curves/stop_signals/watch_submit.done"
)
LOG_PREFIX = "[watch_submit]"


def main() -> int:
    print(f"{LOG_PREFIX} watching {ROOT}", flush=True)
    for _ in range(120):  # ~60 min
        linear = (
            list((ROOT / "linear").glob("emabrats2update_0.9999_*.pt"))
            if (ROOT / "linear").is_dir()
            else []
        )
        aniso = (
            list((ROOT / "anisotropic").glob("emabrats2update_0.9999_*.pt"))
            if (ROOT / "anisotropic").is_dir()
            else []
        )
        print(
            f"{LOG_PREFIX} {time.strftime('%H:%M:%S')} "
            f"linear={len(linear)} anisotropic={len(aniso)}",
            flush=True,
        )
        if linear and aniso:
            print(f"{LOG_PREFIX} FOUND — submitting training-ckpt campaigns", flush=True)
            rc = subprocess.call(
                [sys.executable, str(REPO / "inference" / "submit_training_ckpts_grid.py")],
                cwd=str(REPO),
            )
            FLAG.parent.mkdir(parents=True, exist_ok=True)
            FLAG.write_text(f"rc={rc}\ntime={time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
            print(f"{LOG_PREFIX} submit rc={rc}", flush=True)
            return rc
        time.sleep(30)
    print(f"{LOG_PREFIX} TIMEOUT waiting for synced ckpts", flush=True)
    FLAG.parent.mkdir(parents=True, exist_ok=True)
    FLAG.write_text("timeout\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
