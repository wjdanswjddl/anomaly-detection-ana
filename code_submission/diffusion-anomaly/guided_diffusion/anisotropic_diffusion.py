"""
Anisotropic (spatially-varying) noise diffusion for non-uniform / sparse images.

Provides two classes:

  AnisotropicGaussianDiffusion  —  drop-in replacement for GaussianDiffusion
  SpacedAnisotropicDiffusion    —  drop-in replacement for SpacedDiffusion

Paper reference
---------------
DiffSSC (arxiv 2409.18092, "DiffSSC: Semantic LiDAR Scan Completion using
Denoising Diffusion Probabilistic Models") introduces local noise offsets for
anisotropic data.  The paper's forward process (Eq. 3) is:

    y_t = y_0  +  sqrt(1 - alpha_bar_t) · W · eps

where W is a fixed block-diagonal matrix with per-domain scalar factors
(sigma_p = 1.0 for spatial position, sigma_s = 0.2 for semantic logits).
The model predicts the scaled noise and the training loss (Eq. 4) is:

    L = || sqrt(1 - alpha_bar_t) · W · eps  -  eps_theta(y_t, t) ||²

No extra spatial weighting is applied to the loss; the weighting is implicit
in the magnitude of the target W·eps itself.

Adaptation for 2-D image diffusion
------------------------------------
DiffSSC works on 3-D point clouds with two fixed feature domains.  For 2-D
image grids (LArTPC wire images, medical scans) the relevant anisotropy is
between occupied signal pixels and empty background, not between feature
types.  This implementation adapts the idea as follows:

1. Forward process uses the standard image-DDPM form (signal must decay):
       x_t = sqrt(ab_t) · x_0  +  sqrt(1 - ab_t) · S(x_0) · eps
   rather than DiffSSC's point-cloud form which has no signal decay.

2. The fixed domain matrix W is replaced by a data-derived per-pixel map
   S(x_0) computed from local image statistics (see three modes below).

3. Gaussian smoothing is applied to S to prevent hard artefacts at
   occupied/empty boundaries (not present in DiffSSC).

The training loss matches the paper:

    L = || S · eps  -  eps_theta(x_t, t) ||²

No extra factor of S is applied to the squared error.  The occupied-region
weighting is implicit — those pixels have larger target values and therefore
contribute larger gradient terms naturally.

Three noise modes
-----------------
"binary_mask"
    Occupied pixels (|x| > occupancy_threshold) get S = 1.0.
    Empty pixels get S = empty_noise_fraction (default 0.05).
    Analogous to DiffSSC's two-domain W, adapted for 2-D occupancy.

"signal_proportional"  (recommended for LArTPC / wide dynamic-range data)
    S = |x| / max(|x|), normalised per channel per image.
    S ∈ [empty_noise_fraction, 1]: the brightest pixel gets standard DDPM
    noise; all others get proportionally less.  Empty pixels receive
    empty_noise_fraction noise.

"local_std"
    S = local_std(x) / max(local_std(x)) over patch_size × patch_size
    windows.  S ∈ [empty_noise_fraction, 1].  Captures texture / edge
    structure rather than amplitude peaks.

In all modes S is Gaussian-smoothed and clamped to [empty_noise_fraction, 1].
"""

import math

import numpy as np
import torch as th
import torch.nn.functional as F

from .gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
    _extract_into_tensor,
)
from .nn import mean_flat


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class AnisotropicGaussianDiffusion(GaussianDiffusion):
    """
    Spatially-varying noise diffusion.  Drop-in replacement for GaussianDiffusion.

    Constructor accepts every keyword argument that GaussianDiffusion does,
    plus the extras below.

    Extra Parameters
    ----------------
    noise_mode : str
        "binary_mask" | "signal_proportional" | "local_std".
        Default: "signal_proportional".
    empty_noise_fraction : float
        Floor value for the scale map; prevents empty regions from receiving
        zero noise.  Default: 0.05.
    smoothing_sigma : float
        Sigma (pixels) of the Gaussian blur applied to the scale map to
        soften occupied/empty boundaries.  Set to 0 to disable.
        Default: 2.0.
    occupancy_threshold : float
        Absolute pixel value above which a pixel is "occupied" (used by
        "binary_mask" and as the normalisation floor for
        "signal_proportional").  Default: 0.01.
    local_std_patch_size : int
        Square window size for "local_std" mode.  Default: 8.
    """

    def __init__(
        self,
        *,
        noise_mode: str = "signal_proportional",
        empty_noise_fraction: float = 0.05,
        smoothing_sigma: float = 2.0,
        occupancy_threshold: float = 0.01,
        local_std_patch_size: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.noise_mode = noise_mode
        self.empty_noise_fraction = empty_noise_fraction
        self.smoothing_sigma = smoothing_sigma
        self.occupancy_threshold = occupancy_threshold
        self.local_std_patch_size = local_std_patch_size

    # ------------------------------------------------------------------
    # Scale-map computation
    # ------------------------------------------------------------------

    def _gaussian_smooth(self, x: th.Tensor) -> th.Tensor:
        """
        Depthwise Gaussian blur over the spatial dimensions of x (N, C, H, W).
        Kernel is built on-the-fly so it lives on the correct device and
        handles arbitrary channel counts without caching complications.
        """
        sigma = self.smoothing_sigma
        if sigma <= 0:
            return x

        radius = int(math.ceil(3.0 * sigma))
        size = 2 * radius + 1
        C = x.shape[1]

        coords = th.arange(size, dtype=x.dtype, device=x.device) - radius
        g1d = th.exp(-0.5 * (coords / sigma) ** 2)
        g1d = g1d / g1d.sum()

        kernel2d = g1d[:, None] * g1d[None, :]          # (size, size)
        # Depthwise: (C, 1, size, size) — one filter per channel.
        kernel = (
            kernel2d.unsqueeze(0).unsqueeze(0).expand(C, 1, size, size).contiguous()
        )

        return F.conv2d(x, kernel, padding=radius, groups=C)

    def _compute_noise_scale(self, x_start: th.Tensor) -> th.Tensor:
        """
        Return the spatial noise scale map S for a batch of images.

        Parameters
        ----------
        x_start : Tensor  (N, C, H, W) or (C, H, W)

        Returns
        -------
        scale : Tensor, same shape as x_start, values >= empty_noise_fraction.
        """
        mode = self.noise_mode
        eps_floor = self.empty_noise_fraction

        # The pooling/conv helpers below expect 4D input.  Accept an
        # unbatched (C, H, W) tensor for convenience (validation_plots
        # passes a single image this way).
        unbatched = (x_start.ndim == 3)
        if unbatched:
            x_start = x_start.unsqueeze(0)

        if mode == "binary_mask":
            # Binary: occupied = 1.0, empty = empty_noise_fraction.
            occupied = (x_start.abs() > self.occupancy_threshold).to(x_start.dtype)
            scale = occupied + eps_floor * (1.0 - occupied)

        elif mode == "signal_proportional":
            # S ∝ |x|, normalised per-channel per-image by the channel max
            # so that S ∈ [eps_floor, 1].  The brightest pixel gets standard
            # DDPM noise; all others get proportionally less.
            abs_sig = x_start.abs()                          # (N, C, H, W)
            N, C = abs_sig.shape[:2]
            spatial_dims = abs_sig.shape[2:]
            flat = abs_sig.view(N, C, -1)                    # (N, C, H*W)
            max_sig = flat.max(dim=2, keepdim=True).values   # (N, C, 1)
            max_sig = max_sig.clamp(min=1e-6)
            max_sig = max_sig.view(N, C, *([1] * len(spatial_dims)))
            scale = abs_sig / max_sig
            scale = scale.clamp(min=eps_floor)

        elif mode == "local_std":
            # S ∝ local standard deviation over patch_size x patch_size windows.
            P = self.local_std_patch_size
            # Use explicit asymmetric padding so the output is always the same
            # spatial size as the input, regardless of whether P is even or odd.
            # F.avg_pool2d symmetric padding breaks for even kernel sizes.
            pad_lo = P // 2
            pad_hi = P - 1 - pad_lo  # = pad_lo for odd P, = pad_lo - 1 for even P
            x_sq = x_start ** 2
            x_pad    = F.pad(x_start, (pad_lo, pad_hi, pad_lo, pad_hi))
            x_sq_pad = F.pad(x_sq,    (pad_lo, pad_hi, pad_lo, pad_hi))
            local_mean    = F.avg_pool2d(x_pad,    kernel_size=P, stride=1, padding=0)
            local_mean_sq = F.avg_pool2d(x_sq_pad, kernel_size=P, stride=1, padding=0)
            local_var = (local_mean_sq - local_mean ** 2).clamp(min=0.0)
            local_std_map = local_var.sqrt()

            N, C = local_std_map.shape[:2]
            spatial_dims = local_std_map.shape[2:]
            flat = local_std_map.view(N, C, -1)
            max_std = flat.max(dim=2, keepdim=True).values.clamp(min=1e-6)
            max_std = max_std.view(N, C, *([1] * len(spatial_dims)))
            scale = local_std_map / max_std
            scale = scale.clamp(min=eps_floor)

        else:
            raise ValueError(
                f"Unknown noise_mode {self.noise_mode!r}. "
                "Choose 'binary_mask', 'signal_proportional', or 'local_std'."
            )

        # Smooth to remove hard occupied/empty edges.
        scale = self._gaussian_smooth(scale)

        # Re-clamp: blurring can drag boundary values below the floor.
        scale = scale.clamp(min=eps_floor)

        if unbatched:
            scale = scale.squeeze(0)
        return scale

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x_start: th.Tensor,
        t: th.Tensor,
        noise: th.Tensor = None,
    ) -> th.Tensor:
        """
        Anisotropic forward process q(x_t | x_0).

        Noise is scaled by S(x_0) before injection:
            x_t = sqrt(ab_t) * x_0  +  sqrt(1 - ab_t) * S(x_0) * eps
        """
        if noise is None:
            noise = th.randn_like(x_start)
        scale = self._compute_noise_scale(x_start)
        scaled_noise = scale * noise
        # Delegate to the standard formula with the pre-scaled noise.
        return GaussianDiffusion.q_sample(self, x_start, t, noise=scaled_noise)

    # ------------------------------------------------------------------
    # Training losses
    # ------------------------------------------------------------------

    def training_losses(
        self,
        model,
        x_start: th.Tensor,
        t: th.Tensor,
        classifier=None,
        model_kwargs=None,
        noise: th.Tensor = None,
        pixel_wgt: th.Tensor = None,
    ):
        """
        Anisotropic training losses (DiffSSC Eq. 4, adapted for 2-D images).

        The epsilon target is the *scaled* noise S·eps rather than raw eps,
        matching the anisotropic forward process.  Following the paper, no
        additional spatial weight is applied to the squared error — the
        occupied-region weighting is implicit in the larger target magnitude.

        Returns
        -------
        (terms, target, model_output) — same tuple as GaussianDiffusion.training_losses.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        # Compute the spatial scale map and the scaled noise the model must predict.
        scale = self._compute_noise_scale(x_start)
        scaled_noise = scale * noise  # S · eps

        # Build x_t with the scaled noise.  We call GaussianDiffusion.q_sample
        # directly to bypass our own override — otherwise the noise would be
        # scaled a second time.
        x_t = GaussianDiffusion.q_sample(self, x_start, t, noise=scaled_noise)

        terms = {}

        if self.loss_type in (LossType.KL, LossType.RESCALED_KL):
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps

        elif self.loss_type in (LossType.MSE, LossType.RESCALED_MSE):
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in (ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE):
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    terms["vb"] *= self.num_timesteps / 1000.0

            # The epsilon target is the *scaled* noise so the model learns
            # to predict S · eps, consistent with the anisotropic forward pass.
            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: scaled_noise,
            }[self.model_mean_type]

            assert model_output.shape == target.shape == x_start.shape

            # Loss follows DiffSSC Eq. 4: unweighted squared error against the
            # scaled target.  Occupied pixels naturally dominate the gradient
            # because their target values (S·eps) are larger in magnitude.
            err = (target - model_output) ** 2

            if pixel_wgt is not None:
                err = err * pixel_wgt

            terms["mse"] = mean_flat(err)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]

        else:
            raise NotImplementedError(self.loss_type)

        return (terms, target, model_output)


# ---------------------------------------------------------------------------
# Spaced (fast-sampling) variant
# ---------------------------------------------------------------------------


class _WrappedModel:
    """
    Remaps compressed timestep indices back to the original full schedule
    before calling the model.  Identical to the private class in respace.py —
    duplicated here so this module is self-contained.
    """

    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)


class SpacedAnisotropicDiffusion(AnisotropicGaussianDiffusion):
    """
    Anisotropic diffusion that can skip steps in the original schedule.

    Drop-in replacement for SpacedDiffusion when the underlying diffusion
    process is AnisotropicGaussianDiffusion.  The timestep-spacing logic is
    identical to SpacedDiffusion; the anisotropic noise logic is inherited
    unchanged from AnisotropicGaussianDiffusion.

    Parameters
    ----------
    use_timesteps : collection of int
        Which timesteps from the original full schedule to retain.
    **kwargs
        All keyword arguments for AnisotropicGaussianDiffusion (including
        the anisotropic-specific ones: noise_mode, empty_noise_fraction,
        smoothing_sigma, occupancy_threshold, local_std_patch_size).
    """

    # Keys consumed by AnisotropicGaussianDiffusion beyond GaussianDiffusion.
    _ANISO_KEYS = frozenset({
        "noise_mode",
        "empty_noise_fraction",
        "smoothing_sigma",
        "occupancy_threshold",
        "local_std_patch_size",
    })

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        # Build a full-schedule instance to read alphas_cumprod.
        base_diffusion = AnisotropicGaussianDiffusion(**kwargs)

        # Compute the compressed beta schedule by striding alpha_cumprod.
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1.0 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)

        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Model wrapping — identical pattern to SpacedDiffusion
    # ------------------------------------------------------------------

    def p_mean_variance(self, model, *args, **kwargs):
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, *args, **kwargs):
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score2(self, cond_fn, *args, **kwargs):
        return super().condition_score2(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is delegated to the wrapped model.
        return t
