"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.

    Single-process training on EAF MIG: do NOT overwrite CUDA_VISIBLE_DEVICES —
    Jupyter already assigns the MIG UUID/index. Forcing '0' triggers NVML
    asserts in CUDACachingAllocator on MIG slices.
    """
    if dist.is_initialized():
        return

    # Preserve kernel/job GPU assignment (MIG UUID, etc.).
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # MIG + NCCL is fragile for world_size=1; gloo is enough for single process.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if th.cuda.is_available() and world_size > 1:
        backend = "nccl"
    else:
        backend = "gloo"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["RANK"] = os.environ.get("RANK", "0")
    os.environ["WORLD_SIZE"] = str(world_size)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    print("port2", port)
    print(
        "setup_dist:",
        f"backend={backend}",
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        f"cuda_available={th.cuda.is_available()}",
        f"device_count={th.cuda.device_count() if th.cuda.is_available() else 0}",
    )
    if th.cuda.is_available():
        print("setup_dist: device0=", th.cuda.get_device_name(0))
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    mpigetrank = 0
    if mpigetrank == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
