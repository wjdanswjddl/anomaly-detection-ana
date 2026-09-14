import numpy as np
import torch as th

from guided_diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
    get_named_beta_schedule,
)


def _make_diffusion(num_steps=1000):
    betas = get_named_beta_schedule("linear", num_steps)
    return GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )


def test_q_sample_deterministic_with_fixed_noise():
    d = _make_diffusion()
    x_start = th.randn(2, 1, 8, 8)
    noise = th.randn_like(x_start)
    t = th.tensor([100, 500])
    out_a = d.q_sample(x_start, t, noise=noise)
    out_b = d.q_sample(x_start, t, noise=noise)
    assert th.equal(out_a, out_b)


def test_q_sample_shape_preserved():
    d = _make_diffusion()
    x_start = th.randn(3, 1, 16, 16)
    t = th.zeros(3, dtype=th.long)
    out = d.q_sample(x_start, t)
    assert out.shape == x_start.shape


def test_q_sample_at_t0_close_to_xstart():
    d = _make_diffusion()
    x_start = th.randn(2, 1, 8, 8)
    noise = th.zeros_like(x_start)
    t = th.zeros(2, dtype=th.long)
    out = d.q_sample(x_start, t, noise=noise)
    expected_scale = float(np.sqrt(d.alphas_cumprod[0]))
    assert th.allclose(out, x_start * expected_scale, atol=1e-5)
