from guided_diffusion.gaussian_diffusion import (
    LossType,
    ModelMeanType,
    ModelVarType,
    get_named_beta_schedule,
)
from guided_diffusion.respace import SpacedDiffusion, space_timesteps


def test_space_timesteps_section_count():
    used = space_timesteps(1000, [25])
    assert len(used) == 25
    assert all(0 <= t < 1000 for t in used)


def test_space_timesteps_ddim_prefix():
    used = space_timesteps(1000, "ddim25")
    assert len(used) == 25


def test_spaced_diffusion_num_timesteps_matches():
    betas = get_named_beta_schedule("linear", 1000)
    used = space_timesteps(1000, [25])
    sd = SpacedDiffusion(
        use_timesteps=used,
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )
    assert sd.num_timesteps == 25
    assert len(sd.timestep_map) == 25
