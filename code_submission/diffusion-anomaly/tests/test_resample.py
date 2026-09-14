import torch as th

from guided_diffusion.resample import UniformSampler


def test_uniform_sampler_returns_valid_indices_and_weights():
    sampler = UniformSampler(diffusion=None, maxt=100)
    indices, weights = sampler.sample(8, "cpu")
    assert indices.shape == (8,)
    assert weights.shape == (8,)
    assert ((indices >= 0) & (indices < 100)).all()
    assert (weights > 0).all()


def test_uniform_sampler_weights_uniform():
    sampler = UniformSampler(diffusion=None, maxt=50)
    _, weights = sampler.sample(16, "cpu")
    assert th.allclose(weights, weights[0] * th.ones_like(weights))
