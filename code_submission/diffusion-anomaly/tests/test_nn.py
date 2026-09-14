import torch as th

from guided_diffusion.nn import timestep_embedding, update_ema


def test_update_ema_half_blend():
    target = [th.full((3,), 2.0)]
    source = [th.full((3,), 4.0)]
    update_ema(target, source, rate=0.5)
    assert th.allclose(target[0], th.full((3,), 3.0))


def test_update_ema_rate_one_keeps_target():
    target = [th.full((4,), 7.0)]
    source = [th.full((4,), 11.0)]
    update_ema(target, source, rate=1.0)
    assert th.allclose(target[0], th.full((4,), 7.0))


def test_timestep_embedding_shape():
    t = th.arange(5)
    emb = timestep_embedding(t, dim=16)
    assert emb.shape == (5, 16)


def test_timestep_embedding_odd_dim():
    t = th.arange(3)
    emb = timestep_embedding(t, dim=15)
    assert emb.shape == (3, 15)
