import argparse

import h5py
import numpy as np
import torch as th

from guided_diffusion.autoencoder import AEObjective, CAEModel, VAEModel
from guided_diffusion.image_datasets import DECONV_SIGNAL_SCALE, ImageDataset
from guided_diffusion.resample import UniformSampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    autoencoder_defaults,
    create_autoencoder,
)

# Small configuration used throughout: 32x32 images, 3 encoder stages
# (32 -> 4 spatial bottleneck), 16-dim latent.
SMALL = dict(image_size=32, in_channels=1, latent_dim=16, hidden_dims="16,32,32")


def _small_vae(**kwargs):
    return VAEModel(**{**SMALL, **kwargs})


def _small_cae(**kwargs):
    return CAEModel(**{**SMALL, **kwargs})


def test_vae_forward_shapes():
    model = _small_vae()
    model.eval()
    x = th.randn(2, 1, 32, 32)
    with th.no_grad():
        recon, mu, logvar = model(x)
    assert recon.shape == (2, 1, 32, 32)
    assert mu.shape == (2, 16)
    assert logvar.shape == (2, 16)
    # tanh output activation
    assert recon.min() >= -1.0 and recon.max() <= 1.0


def test_cae_forward_shapes_spatial():
    model = _small_cae(spatial_latent=True)
    model.eval()
    x = th.randn(2, 1, 32, 32)
    with th.no_grad():
        recon = model(x)
    assert recon.shape == (2, 1, 32, 32)
    assert recon.min() >= -1.0 and recon.max() <= 1.0


def test_cae_forward_shapes_dense():
    model = _small_cae(spatial_latent=False)
    model.eval()
    x = th.randn(2, 1, 32, 32)
    with th.no_grad():
        recon = model(x)
    assert recon.shape == (2, 1, 32, 32)


def test_vae_loss_terms():
    model = _small_vae()
    x = th.randn(2, 1, 32, 32)
    # A perfect reconstruction with the prior posterior (mu=0, logvar=0)
    # has zero mse and zero KLD.
    terms = model.loss_terms(x, (x, th.zeros(2, 16), th.zeros(2, 16)))
    assert terms["loss"].shape == (2,)
    assert th.allclose(terms["mse"], th.zeros(2))
    assert th.allclose(terms["kld"], th.zeros(2))
    assert th.allclose(terms["loss"], th.zeros(2))

    # An imperfect reconstruction has a finite, positive loss.
    model.eval()
    with th.no_grad():
        terms = model.loss_terms(x, model(x))
    assert th.isfinite(terms["loss"]).all()
    assert (terms["loss"] > 0).all()


def test_cae_ssim_loss_terms():
    model = _small_cae(ssim_weight=0.5)
    x = th.randn(2, 1, 32, 32).clamp(-1, 1)
    # Perfect reconstruction: l2 = 0 and ssim term = 1 - SSIM(x, x) = 0.
    terms = model.loss_terms(x, x)
    assert th.allclose(terms["l2"], th.zeros(2))
    assert th.allclose(terms["ssim"], th.zeros(2), atol=1e-5)

    model.eval()
    with th.no_grad():
        terms = model.loss_terms(x, model(x))
    assert set(terms.keys()) == {"loss", "l2", "ssim"}
    assert th.isfinite(terms["loss"]).all()


def test_pixel_weight_scaling():
    x = th.randn(2, 1, 32, 32)
    for model in (_small_vae(), _small_cae()):
        model.eval()
        with th.no_grad():
            out = model(x)
            terms1 = model.loss_terms(x, out)
            terms2 = model.loss_terms(x, out, pixel_wgt=2 * th.ones_like(x))
        key = "mse" if isinstance(model, VAEModel) else "l2"
        assert th.allclose(terms2[key], 2 * terms1[key])


def test_reconstruct_and_anomaly_map():
    x = th.randn(2, 1, 32, 32)
    for model in (_small_vae(), _small_cae()):
        model.eval()
        with th.no_grad():
            r1 = model.reconstruct(x)
            r2 = model.reconstruct(x)
            amap = model.anomaly_map(x)
        # Deterministic in eval mode (the VAE decodes the posterior mean).
        assert th.allclose(r1, r2)
        assert amap.shape == x.shape
        assert th.allclose(amap, x - r1)


def test_ae_objective_contract():
    objective = AEObjective()
    assert objective.num_timesteps == 1

    sampler = UniformSampler(objective, maxt=1)
    t, weights = sampler.sample(4, "cpu")
    assert (t == 0).all()
    assert th.allclose(weights, th.ones(4))

    model = _small_cae()
    model.eval()
    x = th.randn(2, 1, 32, 32)
    with th.no_grad():
        result = objective.training_losses(model, x, t[:2])
    assert len(result) == 3
    terms, target, output = result
    assert terms["loss"].shape == (2,)
    assert target.shape == x.shape
    assert output.shape == x.shape


def test_autoencoder_defaults_argparser_roundtrip():
    for ae_type in ("vae", "cae"):
        parser = argparse.ArgumentParser()
        add_dict_to_argparser(parser, autoencoder_defaults())
        args = parser.parse_args(
            [
                "--ae_type", ae_type,
                "--image_size", "32",
                "--ae_hidden_dims", "16,32,32",
                "--ae_latent_dim", "16",
                "--spatial_latent", "False",
            ]
        )
        model = create_autoencoder(**args_to_dict(args, autoencoder_defaults().keys()))
        expected = VAEModel if ae_type == "vae" else CAEModel
        assert isinstance(model, expected)
        model.eval()
        x = th.randn(2, 1, 32, 32)
        with th.no_grad():
            recon = model.reconstruct(x)
        assert recon.shape == x.shape


def test_bad_geometry_raises():
    # 3 encoder stages require image_size divisible by 8.
    try:
        VAEModel(image_size=30, in_channels=1, latent_dim=16, hidden_dims="16,32,32")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for indivisible image_size")


def test_ae_smoke_train_step(tmp_npz):
    path, n, h, w = tmp_npz
    dataset = ImageDataset(h, str(path.parent))
    it = iter(dataset)
    arrs = []
    for _ in range(2):
        arr, out_dict = next(it)
        arrs.append(th.from_numpy(np.array(arr)))
        assert "weight" in out_dict and "pixel_weight" in out_dict
    batch = th.stack(arrs)

    objective = AEObjective()
    for model in (_small_vae(image_size=h), _small_cae(image_size=h)):
        opt = th.optim.AdamW(model.parameters(), lr=1e-4)
        t = th.zeros(batch.shape[0], dtype=th.long)
        terms, _, _ = objective.training_losses(model, batch, t)
        loss = terms["loss"].mean()
        assert th.isfinite(loss)
        loss.backward()
        opt.step()
        # Loss is still finite after an optimizer step.
        terms, _, _ = objective.training_losses(model, batch, t)
        assert th.isfinite(terms["loss"].mean())


def test_h5_deconvolved_signal_loading(tmp_path):
    # ICARUS DNN-ROI-style file: per-event groups with deconvolved_signal
    # and true_number_electrons. 64x96 at resolution 32 -> 6 tiles.
    rng = np.random.default_rng(0)
    path = tmp_path / "icarus.h5"
    with h5py.File(path, "w") as f:
        for ev in range(2):
            g = f.create_group(f"event_{ev}")
            g.create_dataset(
                "deconvolved_signal",
                data=rng.standard_normal((64, 96)).astype(np.float16),
            )
            g.create_dataset(
                "true_number_electrons",
                data=rng.uniform(0, 1000, size=(64, 96)).astype(np.float16),
            )

    dataset = ImageDataset(32, str(tmp_path))
    it = iter(dataset)
    for _ in range(12):  # 2 events x 6 tiles
        arr, out_dict = next(it)
        assert arr.shape == (1, 32, 32)
        assert arr.dtype == np.float32
        assert arr.min() >= -1.0 and arr.max() <= 1.0
        assert np.isfinite(out_dict["weight"])
        assert out_dict["pixel_weight"].shape == (1, 32, 32)

    # The scale constant matches what the loader applied: an all-ones signal
    # would load as 1/DECONV_SIGNAL_SCALE.
    assert DECONV_SIGNAL_SCALE > 0


def test_filelist_loading(tmp_npz, tmp_path):
    # A .txt data path is treated as a file list (one path/URL per line),
    # bypassing directory globbing — required when data is on xrootd.
    path, n, h, w = tmp_npz
    listfile = tmp_path / "files.txt"
    listfile.write_text(f"{path}\n\n")

    dataset = ImageDataset(h, str(listfile))
    assert dataset.local_images == [str(path)]
    arr, out_dict = next(iter(dataset))
    assert arr.shape == (1, h, w)
    assert "weight" in out_dict and "pixel_weight" in out_dict


def test_vae_sample_shapes():
    model = _small_vae()
    model.eval()
    with th.no_grad():
        g = th.Generator().manual_seed(0)
        imgs = model.sample(3, device="cpu", generator=g)
    assert imgs.shape == (3, 1, 32, 32)
    assert imgs.min() >= -1.0 and imgs.max() <= 1.0
    # Seeded sampling is reproducible.
    with th.no_grad():
        g = th.Generator().manual_seed(0)
        imgs2 = model.sample(3, device="cpu", generator=g)
    assert th.allclose(imgs, imgs2)


def test_cae_latent_roundtrip_and_sample():
    x = th.randn(2, 1, 32, 32)
    for spatial in (True, False):
        model = _small_cae(spatial_latent=spatial)
        model.eval()
        with th.no_grad():
            z = model.encode_latent(x)
            assert z.shape[1:] == model.latent_shape
            assert th.allclose(model.decode_latent(z), model(x))
            imgs = model.sample(3, latent_scale=0.5, device="cpu")
        assert imgs.shape == (3, 1, 32, 32)
        assert imgs.min() >= -1.0 and imgs.max() <= 1.0
    # Spatial: 16-dim latent over a 4x4 map -> max(8, 16 // 16) = 8 channels.
    assert _small_cae(spatial_latent=True).latent_shape == (8, 4, 4)
    assert _small_cae(spatial_latent=False).latent_shape == (16,)
