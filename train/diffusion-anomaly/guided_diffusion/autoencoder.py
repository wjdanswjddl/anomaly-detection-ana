"""
Convolutional autoencoder (CAE) and variational autoencoder (VAE) anomaly
detectors for LArTPC images.

Both models are trained on nominal (anomaly-free) images only. At inference
time an input image is passed through the bottleneck and reconstructed; the
signed residual

    anomaly_map(x) = x - reconstruct(x)

is the anomaly/saliency map, matching the convention used by the diffusion
round-trip reconstruction in the comparison notebooks (plotted with
cmap="bwr", vmin/vmax = +/-0.1).

Architectures and methods follow standard, well-cited references (BibTeX in
CITATIONS.md at the repository root):

- VAEModel: the "vanilla" convolutional VAE of Kingma & Welling,
  "Auto-Encoding Variational Bayes" (arXiv:1312.6114), with the widely used
  reference implementation AntixK/PyTorch-VAE (models/vanilla_vae.py,
  Apache-2.0) as the architectural template: stride-2 Conv/BatchNorm/LeakyReLU
  encoder stages, fully-connected mu/logvar heads, mirrored ConvTranspose
  decoder, tanh output. Anomaly scoring by deterministic mean reconstruction
  follows An & Cho, "Variational Autoencoder based Anomaly Detection using
  Reconstruction Probability" (2015).

- CAEModel: the convolutional autoencoder baseline of Bergmann et al.,
  "Improving Unsupervised Defect Segmentation by Applying Structural
  Similarity to Autoencoders" (arXiv:1807.02011), with an optional SSIM loss
  term as proposed there, and an optional dense (fully-connected) bottleneck
  in the style of the autoencoder baselines surveyed by Baur et al.
  (arXiv:2004.03271).

The AEObjective shim at the bottom exposes the models to the existing
TrainLoop through the same training_losses() contract as GaussianDiffusion,
so training inherits EMA, checkpointing, microbatching, and logging
unchanged.
"""

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .losses import normal_kl
from .fp16_util import convert_module_to_f16, convert_module_to_f32


def _parse_hidden_dims(hidden_dims):
    if isinstance(hidden_dims, str):
        return [int(h) for h in hidden_dims.split(",") if h]
    return list(hidden_dims)


def mean_flat_weighted(x, pixel_wgt=None):
    """
    Mean over all non-batch dimensions, optionally weighted per-pixel.
    pixel_wgt is normalized to mean 1 by the dataset, so the weighted and
    unweighted losses have the same scale.
    """
    if pixel_wgt is not None:
        x = x * pixel_wgt
    return x.mean(dim=list(range(1, len(x.shape))))


def _final_activation(name):
    if name == "tanh":
        return nn.Tanh()
    if name in ("none", "identity", ""):
        return nn.Identity()
    raise ValueError(f"Unknown final_activation: {name}")


class ConvEncoder(nn.Module):
    """Stride-2 Conv2d + BatchNorm + LeakyReLU stages (AntixK vanilla VAE)."""

    def __init__(self, in_channels, hidden_dims):
        super().__init__()
        layers = []
        c_in = in_channels
        for h in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(c_in, h, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h),
                    nn.LeakyReLU(),
                )
            )
            c_in = h
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class ConvDecoder(nn.Module):
    """Mirrored ConvTranspose2d + BatchNorm + LeakyReLU stages."""

    def __init__(self, hidden_dims, out_channels, final_activation="tanh"):
        # hidden_dims here is the encoder's list; it is reversed internally.
        super().__init__()
        rev = list(reversed(hidden_dims))
        layers = []
        for i in range(len(rev) - 1):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        rev[i], rev[i + 1],
                        kernel_size=3, stride=2, padding=1, output_padding=1,
                    ),
                    nn.BatchNorm2d(rev[i + 1]),
                    nn.LeakyReLU(),
                )
            )
        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    rev[-1], rev[-1],
                    kernel_size=3, stride=2, padding=1, output_padding=1,
                ),
                nn.BatchNorm2d(rev[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(rev[-1], out_channels, kernel_size=3, padding=1),
                _final_activation(final_activation),
            )
        )
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


def _check_geometry(image_size, hidden_dims):
    reduction = 2 ** len(hidden_dims)
    if image_size % reduction != 0 or image_size // reduction < 1:
        raise ValueError(
            f"image_size={image_size} must be divisible by "
            f"2**len(hidden_dims)={reduction} with a bottleneck of at least "
            f"1x1; got {len(hidden_dims)} encoder stages."
        )
    return image_size // reduction


class _AEBase(nn.Module):
    """Shared reconstruct/anomaly_map/fp16 API for both autoencoders."""

    def reconstruct(self, x):
        raise NotImplementedError

    def anomaly_map(self, x):
        return x - self.reconstruct(x)

    # MixedPrecisionTrainer calls these when use_fp16=True. fp16 is untested
    # for the autoencoders (BatchNorm is kept in fp32 by the converters);
    # the flag scripts pin --use_fp16 False.
    def convert_to_fp16(self):
        self.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.apply(convert_module_to_f32)


class VAEModel(_AEBase):
    """
    Vanilla convolutional VAE (Kingma & Welling arXiv:1312.6114; architecture
    per AntixK/PyTorch-VAE models/vanilla_vae.py, adapted to 1-channel
    LArTPC images).
    """

    def __init__(
        self,
        image_size,
        in_channels=1,
        latent_dim=512,
        hidden_dims="32,64,128,256,512,512,512",
        kld_weight=1e-4,
        final_activation="tanh",
    ):
        super().__init__()
        hidden_dims = _parse_hidden_dims(hidden_dims)
        self.image_size = image_size
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.kld_weight = kld_weight
        self.bottleneck_size = _check_geometry(image_size, hidden_dims)
        self.bottleneck_channels = hidden_dims[-1]

        flat = self.bottleneck_channels * self.bottleneck_size ** 2
        self.encoder = ConvEncoder(in_channels, hidden_dims)
        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_var = nn.Linear(flat, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, flat)
        self.decoder = ConvDecoder(hidden_dims, in_channels, final_activation)

    def encode(self, x):
        h = th.flatten(self.encoder(x), start_dim=1)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, logvar):
        std = th.exp(0.5 * logvar)
        return mu + std * th.randn_like(std)

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(
            -1, self.bottleneck_channels, self.bottleneck_size, self.bottleneck_size
        )
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_terms(self, x, output, pixel_wgt=None):
        recon, mu, logvar = output
        mse = mean_flat_weighted((x - recon) ** 2, pixel_wgt)
        # KLD against the standard normal prior, per-sample, normalized by the
        # number of pixels so kld_weight is resolution-stable (AntixK's M_N
        # scaling plays the same role).
        kld = normal_kl(mu, logvar, 0.0, 0.0).sum(dim=1) / x[0].numel()
        return {"loss": mse + self.kld_weight * kld, "mse": mse, "kld": kld}

    def reconstruct(self, x):
        # Deterministic mean reconstruction (An & Cho 2015).
        mu, _ = self.encode(x)
        return self.decode(mu)

    def sample(self, num_samples, device=None, generator=None):
        """
        Generate images from the prior: decode z ~ N(0, I). Call in eval mode.
        """
        if device is None:
            device = next(self.parameters()).device
        z = th.randn(num_samples, self.latent_dim, device=device, generator=generator)
        return self.decode(z)


class CAEModel(_AEBase):
    """
    Convolutional autoencoder (Bergmann et al. arXiv:1807.02011). Default is
    a fully-convolutional spatial bottleneck; spatial_latent=False uses a
    dense bottleneck in the style of the Baur et al. (arXiv:2004.03271) AE
    baselines. Optional SSIM loss term per Bergmann et al.
    """

    def __init__(
        self,
        image_size,
        in_channels=1,
        latent_dim=512,
        hidden_dims="32,64,128,256,512,512,512",
        spatial_latent=True,
        ssim_weight=0.0,
        final_activation="tanh",
    ):
        super().__init__()
        hidden_dims = _parse_hidden_dims(hidden_dims)
        self.image_size = image_size
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.spatial_latent = spatial_latent
        self.ssim_weight = ssim_weight
        self.bottleneck_size = _check_geometry(image_size, hidden_dims)
        self.bottleneck_channels = hidden_dims[-1]

        self.encoder = ConvEncoder(in_channels, hidden_dims)
        if spatial_latent:
            # 1x1-conv bottleneck; latent_dim distributed over the spatial map
            latent_channels = max(8, latent_dim // self.bottleneck_size ** 2)
            self.bottleneck = nn.Sequential(
                nn.Conv2d(self.bottleneck_channels, latent_channels, kernel_size=1),
                nn.LeakyReLU(),
                nn.Conv2d(latent_channels, self.bottleneck_channels, kernel_size=1),
                nn.LeakyReLU(),
            )
            # The latent code is the (pre-activation) output of the first conv.
            self._latent_split = 1
            self.latent_shape = (latent_channels, self.bottleneck_size, self.bottleneck_size)
        else:
            flat = self.bottleneck_channels * self.bottleneck_size ** 2
            self.bottleneck = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat, latent_dim),
                nn.LeakyReLU(),
                nn.Linear(latent_dim, flat),
                nn.LeakyReLU(),
                nn.Unflatten(
                    1,
                    (self.bottleneck_channels, self.bottleneck_size, self.bottleneck_size),
                ),
            )
            # The latent code is the (pre-activation) output of the first Linear.
            self._latent_split = 2
            self.latent_shape = (latent_dim,)
        self.decoder = ConvDecoder(hidden_dims, in_channels, final_activation)

    def encode_latent(self, x):
        """Latent code at the narrowest point of the bottleneck."""
        return self.bottleneck[: self._latent_split](self.encoder(x))

    def decode_latent(self, z):
        """Inverse of encode_latent: decode_latent(encode_latent(x)) == forward(x)."""
        return self.decoder(self.bottleneck[self._latent_split :](z))

    def forward(self, x):
        return self.decode_latent(self.encode_latent(x))

    def sample(self, num_samples, latent_scale=1.0, device=None, generator=None):
        """
        Decode random latent codes z ~ N(0, latent_scale**2 I). Unlike the VAE,
        the CAE has no prior over its latent space, so this is only a probe of
        what the decoder produces from noise, not generative sampling. Call in
        eval mode.
        """
        if device is None:
            device = next(self.parameters()).device
        z = latent_scale * th.randn(
            num_samples, *self.latent_shape, device=device, generator=generator
        )
        return self.decode_latent(z)

    def loss_terms(self, x, output, pixel_wgt=None):
        recon = output
        l2 = mean_flat_weighted((x - recon) ** 2, pixel_wgt)
        terms = {"l2": l2}
        loss = l2
        if self.ssim_weight > 0:
            ssim_term = 1.0 - ssim(x, recon, data_range=2.0)
            terms["ssim"] = ssim_term
            loss = loss + self.ssim_weight * ssim_term
        terms["loss"] = loss
        return terms

    def reconstruct(self, x):
        return self.forward(x)


def _gaussian_window(window_size, sigma, channels, device, dtype):
    coords = th.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = th.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    window = (g.t() @ g).unsqueeze(0).unsqueeze(0)
    return window.expand(channels, 1, window_size, window_size).contiguous()


def ssim(img1, img2, window_size=11, sigma=1.5, data_range=2.0):
    """
    Per-sample mean structural similarity (Wang et al. 2004), as used for
    autoencoder anomaly segmentation by Bergmann et al. (arXiv:1807.02011).
    Returns a tensor of shape (N,). data_range=2.0 for images in [-1, 1].
    """
    channels = img1.shape[1]
    window = _gaussian_window(window_size, sigma, channels, img1.device, img1.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return mean_flat_weighted(ssim_map)


class AEObjective:
    """
    Adapter exposing the autoencoders to TrainLoop through the same
    training_losses() contract as GaussianDiffusion. The timestep t is
    ignored (use UniformSampler(objective, maxt=1), which always yields t=0
    with unit weights).
    """

    # log_loss_dict buckets losses by t/num_timesteps; with maxt=1 all
    # entries land in quartile 0.
    num_timesteps = 1

    def training_losses(
        self, model, x_start, t,
        classifier=None, model_kwargs=None, noise=None, pixel_wgt=None,
    ):
        output = model(x_start)
        module = model.module if hasattr(model, "module") else model
        terms = module.loss_terms(x_start, output, pixel_wgt=pixel_wgt)
        recon = output[0] if isinstance(output, tuple) else output
        return terms, x_start, recon
