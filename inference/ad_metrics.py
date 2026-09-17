"""General AD scores for diffusion reconstruction outputs.

Designed for semi-sparse LArTPC patches where high-T reconstructions rewrite
the noise floor (so plain pixel MSE is a weak detector). Metrics are
**domain-agnostic** — coherent noise is one failure mode they should catch,
not a special case baked into the formulas.

Pixel metrics need only ``original`` / ``reconstructed`` arrays (usable offline
on existing ``*_arrays.npz``). Model metrics need the loaded UNet + diffusion.

Primary scalars (per patch)
---------------------------
- ``mse`` — mean squared residual (baseline)
- ``weighted_mse`` — amplitude soft-weight toward the noise floor
- ``noise_region_mse`` — MSE on low-|q| mask (hard noise-region cut)
- ``latent_l2`` — L2 distance of UNet bottleneck embeddings (input vs reco)
- ``denoise_loss`` — diffusion noise-prediction MSE on the **input** at ``t_loss``
- ``posterior_typicality`` — how far embed(input) sits from K stochastic healthy
  counterfactuals (q-sample @ T → DDIM denoise); higher = more anomalous
- ``calibrated_residual`` — z-score of ``noise_region_mse`` vs a healthy null
  (computed in notebooks once healthy scores are available; jobs store the raw
  ``noise_region_mse`` used for calibration)

Grid jobs write ``{stem}_T{T}_{mode}_ad_metrics.npz`` alongside arrays/pickles.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------

METRIC_NAMES = (
    "mse",
    "weighted_mse",
    "noise_region_mse",
    "latent_l2",
    "denoise_loss",
    "posterior_typicality",
)


def _as_n1hw(arr: np.ndarray) -> np.ndarray:
    """Normalize to float32 (N, 1, H, W)."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2:
        return a[np.newaxis, np.newaxis, ...]
    if a.ndim == 3:
        # (N, H, W) or (1, H, W)
        if a.shape[0] in (1,) and a.shape[1] > 8 and a.shape[2] > 8:
            return a[:, np.newaxis, ...]
        return a[:, np.newaxis, ...]
    if a.ndim == 4:
        if a.shape[1] != 1:
            raise ValueError(f"expected C=1, got shape {a.shape}")
        return a
    raise ValueError(f"unexpected array ndim={a.ndim} shape={a.shape}")


def _squeeze_hw(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    while a.ndim > 2:
        a = a.squeeze(0) if a.shape[0] == 1 else a.squeeze(1) if a.shape[1] == 1 else a[0]
    return a


# ---------------------------------------------------------------------------
# Pixel-space metrics (no model)
# ---------------------------------------------------------------------------


def soft_noise_weights(
    original: np.ndarray,
    *,
    ref_percentile: float = 99.0,
    soft_scale: float = 0.25,
    power: float = 2.0,
) -> np.ndarray:
    """Per-pixel weights emphasizing low-|charge| (noise-floor) locations.

    ``w = 1 / (1 + (|x| / (soft_scale * ref))^power)``, then mean-normalized
    so weighted MSE is on a similar numeric scale to plain MSE.
    """
    x = np.asarray(original, dtype=np.float64)
    a = np.abs(x)
    if a.ndim >= 3:
        flat = a.reshape(a.shape[0], -1)
        ref = np.percentile(flat, ref_percentile, axis=1)
        ref = ref.reshape((a.shape[0],) + (1,) * (a.ndim - 1))
    else:
        ref = float(np.percentile(a, ref_percentile))
    ref = np.maximum(ref, 1e-6)
    w = 1.0 / (1.0 + (a / (soft_scale * ref + 1e-12)) ** power)
    axes = tuple(range(1, w.ndim)) if w.ndim > 1 else ()
    if axes:
        w = w / np.maximum(w.mean(axis=axes, keepdims=True), 1e-12)
    else:
        w = w / max(float(w.mean()), 1e-12)
    return w.astype(np.float32)


def noise_region_mask(
    original: np.ndarray,
    *,
    abs_quantile: float = 0.70,
) -> np.ndarray:
    """Boolean mask of low-|q| pixels (per patch), True = noise region."""
    x = np.asarray(original, dtype=np.float64)
    a = np.abs(x)
    if a.ndim >= 3:
        q = np.quantile(a.reshape(a.shape[0], -1), abs_quantile, axis=1)
        q = q.reshape(-1, *([1] * (a.ndim - 1)))
    else:
        q = float(np.quantile(a, abs_quantile))
    return a <= q


def pixel_metric_maps(
    original: np.ndarray,
    reconstructed: np.ndarray,
    *,
    abs_quantile: float = 0.70,
    weight_kwargs: dict | None = None,
) -> dict[str, np.ndarray]:
    """Return residual maps used for scoring / optional visualization."""
    orig = _as_n1hw(original)
    reco = _as_n1hw(reconstructed)
    if orig.shape != reco.shape:
        raise ValueError(f"shape mismatch orig={orig.shape} reco={reco.shape}")
    sq = (orig.astype(np.float64) - reco.astype(np.float64)) ** 2
    w = soft_noise_weights(orig, **(weight_kwargs or {}))
    mask = noise_region_mask(orig, abs_quantile=abs_quantile)
    return {
        "sq_err": sq.astype(np.float32),
        "weighted_sq_err": (w * sq).astype(np.float32),
        "noise_weights": w.astype(np.float32),
        "noise_mask": mask.astype(np.bool_),
    }


def pixel_metric_scores(
    original: np.ndarray,
    reconstructed: np.ndarray,
    *,
    abs_quantile: float = 0.70,
    weight_kwargs: dict | None = None,
) -> dict[str, np.ndarray]:
    """Per-patch scalar pixel metrics. Shapes: (N,)."""
    maps = pixel_metric_maps(
        original, reconstructed, abs_quantile=abs_quantile, weight_kwargs=weight_kwargs
    )
    sq = maps["sq_err"].astype(np.float64)
    wsq = maps["weighted_sq_err"].astype(np.float64)
    mask = maps["noise_mask"]
    axes = tuple(range(1, sq.ndim))
    mse = sq.mean(axis=axes)
    weighted_mse = wsq.mean(axis=axes)
    # noise-region MSE: mean over masked pixels; empty mask → nan
    n = mask.reshape(mask.shape[0], -1).sum(axis=1).astype(np.float64)
    masked_sum = (sq * mask).reshape(sq.shape[0], -1).sum(axis=1)
    noise_region_mse = np.where(n > 0, masked_sum / n, np.nan)
    return {
        "mse": mse.astype(np.float32),
        "weighted_mse": weighted_mse.astype(np.float32),
        "noise_region_mse": noise_region_mse.astype(np.float32),
    }


def calibrate_scores(
    scores: np.ndarray,
    healthy_scores: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Z-score ``scores`` using mean/std of ``healthy_scores`` (finite only)."""
    h = np.asarray(healthy_scores, dtype=np.float64)
    h = h[np.isfinite(h)]
    if h.size < 2:
        return np.full_like(scores, np.nan, dtype=np.float64)
    mu, sig = float(h.mean()), float(h.std())
    sig = max(sig, eps)
    return (np.asarray(scores, dtype=np.float64) - mu) / sig


# ---------------------------------------------------------------------------
# Model-space metrics
# ---------------------------------------------------------------------------


def unet_bottleneck_embedding(model, x, timesteps) -> "th.Tensor":
    """Global-average-pool the UNet middle-block features → (N, C)."""
    import torch as th
    from guided_diffusion.nn import timestep_embedding

    assert (getattr(model, "num_classes", None) is None), "class-cond UNet not supported here"
    emb = model.time_embed(timestep_embedding(timesteps, model.model_channels))
    h = x.type(model.dtype)
    for module in model.input_blocks:
        h = module(h, emb)
    h = model.middle_block(h, emb)
    # (N, C, H', W') → (N, C)
    return h.float().mean(dim=(2, 3))


def latent_l2_scores(
    model,
    original: "th.Tensor",
    reconstructed: "th.Tensor",
    *,
    t_embed: int = 0,
) -> np.ndarray:
    """Per-patch L2 distance between bottleneck embeddings of orig vs reco."""
    import torch as th

    n = original.shape[0]
    t = th.full((n,), int(t_embed), device=original.device, dtype=th.long)
    with th.no_grad():
        e0 = unet_bottleneck_embedding(model, original, t)
        e1 = unet_bottleneck_embedding(model, reconstructed, t)
        d = th.linalg.vector_norm(e0 - e1, dim=1)
    return d.detach().cpu().numpy().astype(np.float32)


def denoise_loss_scores(
    diffusion,
    model,
    x0: "th.Tensor",
    *,
    t_loss: int,
    noise: "th.Tensor | None" = None,
) -> np.ndarray:
    """Per-patch MSE of noise prediction on the **input** at timestep ``t_loss``."""
    import torch as th

    n = x0.shape[0]
    t = th.full((n,), int(t_loss), device=x0.device, dtype=th.long)
    if noise is None:
        noise = th.randn_like(x0)
    with th.no_grad():
        x_t = diffusion.q_sample(x0, t, noise=noise)
        # match training_losses scaling of timesteps
        t_in = diffusion._scale_timesteps(t) if hasattr(diffusion, "_scale_timesteps") else t
        out = model(x_t, t_in)
        c = x0.shape[1]
        if out.shape[1] == 2 * c:
            out = out[:, :c]
        # EPSILON prediction (Gray / our defaults); if predict_xstart, compare to x0
        predict_xstart = bool(getattr(diffusion, "predict_xstart", False))
        if hasattr(diffusion, "model_mean_type"):
            from guided_diffusion.gaussian_diffusion import ModelMeanType

            predict_xstart = diffusion.model_mean_type == ModelMeanType.START_X
        target = x0 if predict_xstart else noise
        err = (out - target) ** 2
        loss = err.reshape(n, -1).mean(dim=1)
    return loss.detach().cpu().numpy().astype(np.float32)


def posterior_typicality_scores(
    diffusion,
    model,
    x0: "th.Tensor",
    *,
    T: int,
    k: int,
    t_embed: int = 0,
    progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance of embed(x0) to the cloud of K stochastic healthy counterfactuals.

    Each counterfactual: ``x_T ~ q(x_T|x0)`` then DDIM denoise back (independent noise).
    Returns ``(scores (N,), embeddings_posterior (N, K, C))``.
    """
    import torch as th

    if k <= 0:
        n = x0.shape[0]
        return (
            np.full((n,), np.nan, dtype=np.float32),
            np.zeros((n, 0, 0), dtype=np.float32),
        )

    n = x0.shape[0]
    t_emb = th.full((n,), int(t_embed), device=x0.device, dtype=th.long)
    with th.no_grad():
        e_in = unet_bottleneck_embedding(model, x0, t_emb)
        e_posts: list = []
        for _ in range(int(k)):
            t = th.full((n,), int(T), device=x0.device, dtype=th.long)
            x_t = diffusion.q_sample(x0, t)
            gen = diffusion.ddim_sample_loop_progressive(
                model, x0.shape, time=int(T), noise=x_t, progress=progress
            )
            reco = list(gen)[-1]["sample"]
            e_posts.append(unet_bottleneck_embedding(model, reco, t_emb))
        # (K, N, C) → (N, K, C)
        stack = th.stack(e_posts, dim=0).permute(1, 0, 2)
        # distance to mean of posterior embeddings
        mu = stack.mean(dim=1)
        scores = th.linalg.vector_norm(e_in - mu, dim=1)
        # also record min distance to any sample (sometimes sharper)
        # keep mean-distance as primary; both useful — store mean in scores
    return (
        scores.detach().cpu().numpy().astype(np.float32),
        stack.detach().cpu().numpy().astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Batch compute + I/O
# ---------------------------------------------------------------------------


def compute_ad_metrics_batch(
    original_np: np.ndarray,
    reconstructed_np: np.ndarray,
    *,
    model=None,
    diffusion=None,
    T: int | None = None,
    t_loss: int | None = None,
    t_embed: int = 0,
    posterior_k: int = 0,
    abs_quantile: float = 0.70,
    device=None,
    progress_posterior: bool = False,
) -> dict[str, Any]:
    """Compute all available metrics for a batch of patches.

    Model metrics are filled with NaN if ``model``/``diffusion`` are omitted.
    """
    orig = _as_n1hw(original_np)
    reco = _as_n1hw(reconstructed_np)
    pix = pixel_metric_scores(orig, reco, abs_quantile=abs_quantile)
    maps = pixel_metric_maps(orig, reco, abs_quantile=abs_quantile)
    n = orig.shape[0]
    out: dict[str, Any] = {
        **pix,
        "latent_l2": np.full((n,), np.nan, dtype=np.float32),
        "denoise_loss": np.full((n,), np.nan, dtype=np.float32),
        "posterior_typicality": np.full((n,), np.nan, dtype=np.float32),
        "emb_input": None,
        "emb_reco": None,
        "emb_posterior": None,
        # Keep maps only in-memory for optional viz; not written to grid NPZ by default
        # (full 512² maps dominate tarball size). Recompute from arrays offline.
        "_maps": maps,
        "metric_names": list(METRIC_NAMES),
        "abs_quantile": float(abs_quantile),
        "t_loss": None,
        "t_embed": int(t_embed),
        "posterior_k": int(posterior_k),
        "T": None if T is None else int(T),
    }

    if model is None or diffusion is None:
        return out

    import torch as th

    from guided_diffusion import dist_util

    dev = device or dist_util.dev()
    x0 = th.tensor(orig, device=dev)
    xhat = th.tensor(reco, device=dev)

    out["latent_l2"] = latent_l2_scores(model, x0, xhat, t_embed=t_embed)
    with th.no_grad():
        t_e = th.full((n,), int(t_embed), device=dev, dtype=th.long)
        out["emb_input"] = (
            unet_bottleneck_embedding(model, x0, t_e).detach().cpu().numpy().astype(np.float32)
        )
        out["emb_reco"] = (
            unet_bottleneck_embedding(model, xhat, t_e).detach().cpu().numpy().astype(np.float32)
        )

    t_use = int(t_loss) if t_loss is not None else (int(T) if T is not None else 200)
    out["t_loss"] = t_use
    out["denoise_loss"] = denoise_loss_scores(diffusion, model, x0, t_loss=t_use)

    if posterior_k > 0 and T is not None and int(T) > 0:
        typ, emb_p = posterior_typicality_scores(
            diffusion,
            model,
            x0,
            T=int(T),
            k=int(posterior_k),
            t_embed=t_embed,
            progress=progress_posterior,
        )
        out["posterior_typicality"] = typ
        out["emb_posterior"] = emb_p
        out["T"] = int(T)

    return out


def save_ad_metrics_npz(path: Path, metrics: dict[str, Any], *, trace: dict | None = None) -> Path:
    """Write compact per-patch AD metrics NPZ."""
    path = Path(path)
    payload: dict[str, Any] = {}
    for name in METRIC_NAMES:
        if name in metrics and metrics[name] is not None:
            payload[name] = np.asarray(metrics[name], dtype=np.float32)
    for key in ("emb_input", "emb_reco", "emb_posterior"):
        if metrics.get(key) is not None:
            payload[key] = np.asarray(metrics[key])
    # small JSON-ish metadata as arrays
    payload["metric_names"] = np.asarray(metrics.get("metric_names", METRIC_NAMES))
    for k in ("abs_quantile", "t_loss", "t_embed", "posterior_k", "T"):
        if metrics.get(k) is not None:
            payload[k] = np.asarray(metrics[k])
    if trace:
        for k, v in trace.items():
            payload[f"trace_{k}"] = np.asarray(v if v is not None else "")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def load_ad_metrics_npz(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def compute_pixel_metrics_from_arrays_npz(arrays_path: Path, **kwargs) -> dict[str, np.ndarray]:
    """Offline pixel metrics from an existing ``*_arrays.npz`` (no model)."""
    z = np.load(arrays_path)
    orig = z["original"]
    reco = z["reconstructed"]
    return pixel_metric_scores(orig, reco, **kwargs)


def file_score_from_patch_scores(patch_scores: np.ndarray, reduction: str = "max") -> float:
    """Reduce per-patch scores to a file-level score."""
    s = np.asarray(patch_scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan")
    if reduction == "max":
        return float(np.max(s))
    if reduction == "mean":
        return float(np.mean(s))
    if reduction == "p95":
        return float(np.percentile(s, 95))
    raise ValueError(f"unknown reduction {reduction!r}")
