import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@pytest.fixture
def tmp_npz(tmp_path):
    """Synthetic SBND-style .npz with reco/truth arrays of shape (N, 1, H, W)."""
    rng = np.random.default_rng(0)
    n, h, w = 4, 32, 32
    reco = rng.standard_normal((n, 1, h, w)).astype(np.float32)
    truth = rng.uniform(0.0, 1.0, size=(n, 1, h, w)).astype(np.float32)
    path = tmp_path / "sample.npz"
    np.savez(path, reco=reco, truth=truth)
    return path, n, h, w
