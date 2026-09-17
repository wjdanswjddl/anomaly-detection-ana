"""
Run VAE/CAE anomaly-detection inference: reconstruct images from a trained
autoencoder checkpoint and save anomaly (saliency) maps.

For each batch this writes batch_%04d.npz with float32 arrays:
    input    (N, C, H, W)  the input images
    reco     (N, C, H, W)  autoencoder reconstructions
    saliency (N, C, H, W)  input - reco  (the anomaly map)
and, with --save_png True, a 3-panel PNG per image (original |
reconstruction | saliency, the latter with cmap="bwr", vmin/vmax=+/-0.1 to
match the comparison-notebook convention).
"""
import os
import sys
import argparse
import numpy as np
import torch as th
import matplotlib.pyplot as plt
sys.path.append("..")
sys.path.append(".")
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.script_util import (
    autoencoder_defaults,
    create_autoencoder,
    args_to_dict,
    add_dict_to_argparser,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()

    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(args.model_path) or ".", "ae_samples")
    os.makedirs(output_dir, exist_ok=True)
    logger.configure(dir=output_dir)

    logger.log("creating autoencoder (%s)..." % args.ae_type)
    model = create_autoencoder(
        **args_to_dict(args, autoencoder_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("creating data loader...")
    datal = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        charge_scale=args.charge_scale,
        class_cond=False,
        deterministic=True,
    )

    # The data loader cycles forever, so a positive batch count is required.
    assert args.num_batches > 0, "--num_batches must be positive"

    ibatch = 0
    while ibatch < args.num_batches:
        batch, cond = next(datal)
        batch = batch.to(dist_util.dev())
        with th.no_grad():
            reco = model.reconstruct(batch)
        saliency = batch - reco

        inp = batch.cpu().numpy().astype(np.float32)
        rec = reco.cpu().numpy().astype(np.float32)
        sal = saliency.cpu().numpy().astype(np.float32)
        paths = np.array(cond.get("path", [""] * inp.shape[0]))

        outfile = os.path.join(output_dir, "batch_%04d.npz" % ibatch)
        np.savez(outfile, input=inp, reco=rec, saliency=sal, paths=paths)
        logger.log("saved %s" % outfile)

        if args.save_png:
            for i in range(inp.shape[0]):
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
                c1 = ax1.imshow(np.squeeze(inp[i]), vmin=-1, vmax=1)
                fig.colorbar(c1, ax=ax1)
                ax1.title.set_text("Original")

                c2 = ax2.imshow(np.squeeze(rec[i]), vmin=-1, vmax=1)
                fig.colorbar(c2, ax=ax2)
                ax2.title.set_text("Reconstruction")

                c3 = ax3.imshow(np.squeeze(sal[i]), cmap="bwr", vmin=-0.1, vmax=0.1)
                fig.colorbar(c3, ax=ax3)
                ax3.title.set_text("Saliency (Original - Reconstruction)")

                plt.savefig(
                    os.path.join(output_dir, "batch_%04d_img_%02d.png" % (ibatch, i)),
                    bbox_inches="tight",
                )
                plt.close(fig)

        ibatch += 1

    logger.log("sampling complete (%i batches)" % ibatch)


def create_argparser():
    defaults = dict(
        model_path="",
        data_dir="",
        output_dir="",
        charge_scale=1.,
        batch_size=4,
        num_batches=1,
        save_png=True,
    )
    defaults.update(autoencoder_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
