"""
Minimal inference demo for the trained SBND VAE anomaly detector.

Two things happen, in order:

1. Generation from random noise: z ~ N(0, I) is decoded into images
   (VAEModel.sample) -> generated.npz, generated.png.
2. Anomaly detection on an example SBND input file: the file is tiled into
   512x512 images exactly as in training, the --num_tiles tiles with the most
   signal are reconstructed, and saliency = input - reconstruction is the
   anomaly map -> anomaly.npz, tile_XX.png (original | reconstruction |
   saliency, bwr +/-0.1).

The model configuration is hard-coded to the trained run
(model_flags_VAE_SBND.sh, /scratch/7DayLifetime/gputnam/training-SBND-VAE/iterA).

Run from the repository root. SBND raw h5 files live in dCache and are read
over xrootd, so the process needs the xrootd POSIX preload and a valid SBND
bearer token (refresh with: htgettoken -a htvaultprod.fnal.gov -i sbnd):

    LD_PRELOAD=/usr/lib64/libXrdPosixPreload.so HDF5_USE_FILE_LOCKING=FALSE \
        python3 scripts/vae_sbnd_demo.py [--model_path ...] [--input ...]

A local .h5 or .npz --input needs no preload.
"""
import argparse
import os
import sys

import numpy as np
import torch as th
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(".")
sys.path.append("..")
from guided_diffusion.image_datasets import load_image_file
from guided_diffusion.script_util import create_autoencoder

# Must match the flags the checkpoint was trained with.
MODEL_CONFIG = dict(
    ae_type="vae",
    image_size=512,
    in_channels=1,
    ae_hidden_dims="32,64,128,256,512,512,512",
    ae_latent_dim=512,
    kld_weight=1e-4,
    spatial_latent=True,  # unused by the VAE
    ssim_weight=0.0,      # unused by the VAE
    final_activation="tanh",
)
DEFAULT_MODEL = "/exp/sbnd/data/users/munjung/anomaly-detection/training/vae/sbnd/ema_0.9999_263000.pt"
# Upstream scratch (often purged): /scratch/7DayLifetime/gputnam/training-SBND-VAE/iterA/ema_0.9999_263000.pt
DEFAULT_INPUT = (
    "root://fndcadoor.fnal.gov:1094//pnfs/fnal.gov/usr/sbnd/scratch/users/munjung/"
    "v10_06_00/raw/h5/86413898_1205/g4-raw-0_3.h5"
)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", default=DEFAULT_MODEL, help="checkpoint (.pt state dict)")
    p.add_argument("--input", default=DEFAULT_INPUT, help="SBND .h5 file (local or root:// URL) or .npz")
    p.add_argument("--output_dir", default="./results/vae-sbnd-demo")
    p.add_argument("--num_samples", type=int, default=4, help="images to generate from noise")
    p.add_argument("--num_tiles", type=int, default=4, help="input tiles to run anomaly detection on")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = create_autoencoder(**MODEL_CONFIG)
    model.load_state_dict(th.load(args.model_path, map_location="cpu", weights_only=True))
    model.to(device).eval()
    print(f"loaded {args.model_path}")

    # --- 1. generation from random noise ---------------------------------
    gen = th.Generator(device=device).manual_seed(args.seed)
    with th.no_grad():
        samples = model.sample(args.num_samples, device=device, generator=gen).cpu().numpy()
    np.savez(os.path.join(args.output_dir, "generated.npz"), samples=samples)
    # Nominal LArTPC images are mostly empty, so samples have small amplitude;
    # plot them on a symmetric scale set by the data rather than the +/-1
    # range used for the real images below.
    vmax = max(float(np.abs(samples).max()), 1e-3)
    fig, axes = plt.subplots(1, args.num_samples, figsize=(4 * args.num_samples, 4), squeeze=False)
    for i, ax in enumerate(axes[0]):
        im = ax.imshow(samples[i, 0], vmin=-vmax, vmax=vmax, cmap="bwr")
        ax.set_title(f"VAE sample {i}")
        fig.colorbar(im, ax=ax)
    fig.savefig(os.path.join(args.output_dir, "generated.png"), bbox_inches="tight")
    plt.close(fig)
    print(
        f"generated {args.num_samples} images (pixel range {samples.min():+.3f} .. {samples.max():+.3f}) "
        f"-> {args.output_dir}/generated.png"
    )

    # --- 2. anomaly detection on an example input --------------------------
    print(f"loading {args.input} ...")
    images, _ = load_image_file(args.input, MODEL_CONFIG["image_size"])
    # Most LArTPC tiles are empty; show the ones with the most signal.
    order = np.argsort(-np.abs(images).sum(axis=(1, 2, 3)))[: args.num_tiles]
    inp = th.from_numpy(images[order]).to(device)
    with th.no_grad():
        reco = model.reconstruct(inp)
    saliency = (inp - reco).cpu().numpy()
    inp, reco = inp.cpu().numpy(), reco.cpu().numpy()
    np.savez(
        os.path.join(args.output_dir, "anomaly.npz"),
        input=inp, reco=reco, saliency=saliency, tile_index=order, source=str(args.input),
    )
    for k, tile in enumerate(order):
        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4))
        for ax, arr, title, kw in (
            (a1, inp[k, 0], f"Original (tile {tile})", dict(vmin=-1, vmax=1)),
            (a2, reco[k, 0], "Reconstruction", dict(vmin=-1, vmax=1)),
            (a3, saliency[k, 0], "Saliency (Original - Reconstruction)", dict(cmap="bwr", vmin=-0.1, vmax=0.1)),
        ):
            fig.colorbar(ax.imshow(arr, **kw), ax=ax)
            ax.set_title(title)
        fig.savefig(os.path.join(args.output_dir, f"tile_{k:02d}.png"), bbox_inches="tight")
        plt.close(fig)
        print(f"tile {tile}: anomaly score (mean squared residual) = {np.mean(saliency[k] ** 2):.3e}")
    print(f"wrote anomaly maps -> {args.output_dir}/anomaly.npz, tile_XX.png")


if __name__ == "__main__":
    main()
