import numpy as np

from guided_diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
    get_named_beta_schedule,
)


def test_linear_schedule_shape_and_range():
    betas = get_named_beta_schedule("linear", 1000)
    assert betas.shape == (1000,)
    assert (betas > 0).all() and (betas < 1).all()
    assert (np.diff(betas) >= 0).all()


def test_cosine_schedule_shape_and_range():
    betas = get_named_beta_schedule("cosine", 1000)
    assert betas.shape == (1000,)
    assert (betas > 0).all() and (betas <= 0.999).all()


def _make_diffusion(num_steps=1000):
    betas = get_named_beta_schedule("linear", num_steps)
    return GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )


def test_alphas_cumprod_decreasing():
    d = _make_diffusion(1000)
    assert d.num_timesteps == 1000
    assert (np.diff(d.alphas_cumprod) <= 0).all()
    assert d.alphas_cumprod[0] < 1.0
    assert d.alphas_cumprod[-1] > 0.0
