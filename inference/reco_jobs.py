import sys

sys.path.append("anomaly-detection/diffusion-anomaly")

import argparse
from tqdm import tqdm

from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    create_gaussian_diffusion
)

from guided_diffusion.resample import UniformSampler
from guided_diffusion import dist_util

from guided_diffusion.fp16_util import *

from guided_diffusion.respace import space_timesteps

import numpy as np
import matplotlib.pyplot as plt
import torch as th

import h5py

import os

import pickle
from guided_diffusion.train_util import visualize


def load_data(filename, threshold=0):
    with h5py.File(str(filename), "r") as f:
        # scale charge by value
        arrs = [f[ev]["raw"][:] for ev in f.keys()]
    allarrs = []
    for arr in arrs:
        nrows, ncols = (512, 512)
        plane_boundaries = [0, 1984, 3968, 5638]
        for planeno, (wlo, whi) in enumerate(zip(plane_boundaries[:-1], plane_boundaries[1:])):
            cscale = [200., 100., 200.][planeno]
            
            planearr = arr[wlo:whi, :]/cscale
            h, w = planearr.shape
            ll = planearr[:(h//nrows)*nrows, :(w//nrows)*nrows].reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols)

            if np.max(ll) < threshold:
                continue
            allarrs.append(ll)
            

    _cache_file = visualize(np.expand_dims(np.concatenate(allarrs), axis=1)).astype(np.float32)
    _cache_true = np.ones((_cache_file.shape[0], 1, nrows, ncols)).astype(np.float32)
    return _cache_file, _cache_true


def noise_frames(diffusion, model, imgs, T):
    # noised frames

    # DDIM
    ddim_noisef = diffusion.ddim_sample_loop_progressive(model, imgs.shape, time=T, noise=imgs, 
                                                reverse=True, progress=True)

    ddim_noise = list(ddim_noisef)

    ddim_noised = ddim_noise[-1]["sample"]

    # Random
    rand_noised = diffusion.q_sample(imgs, th.tensor(T, device=dist_util.dev()))

    return ddim_noised, rand_noised


def reconstruct_frames(diffusion, model, imgs, ddim_noised, rand_noised, T):
    # Reconstruct
    ddim_2_ddim = diffusion.ddim_sample_loop_progressive(model, 
                    imgs.shape, time=T, noise=ddim_noised, progress=True)

    rand_2_ddim = diffusion.ddim_sample_loop_progressive(model, 
                    imgs.shape, time=T, noise=rand_noised, progress=True)

    ddim_2_ddpm = diffusion.p_sample_loop_progressive(model, 
                    imgs.shape, time=T, noise=ddim_noised, progress=True)

    rand_2_ddpm = diffusion.p_sample_loop_progressive(model, 
                    imgs.shape, time=T, noise=rand_noised, progress=True)


    ddim_2_ddim_reco = list(ddim_2_ddim)[-1]["sample"]
    rand_2_ddim_reco = list(rand_2_ddim)[-1]["sample"]
    ddim_2_ddpm_reco = list(ddim_2_ddpm)[-1]["sample"]
    rand_2_ddpm_reco = list(rand_2_ddpm)[-1]["sample"]

    recos = [
        ddim_2_ddim_reco,
        rand_2_ddim_reco,
        ddim_2_ddpm_reco,
        rand_2_ddpm_reco
    ]

    return recos


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_base_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=3)

    runargs = parser.parse_args()

    input_base_dir = runargs.input_base_dir
    save_dir = runargs.save_dir

    threshold = runargs.threshold

    batch_size = runargs.batch_size


    th.set_grad_enabled(False)

    args = model_and_diffusion_defaults()
    diffusion_args = diffusion_defaults()

    # MODEL
    args["image_size"] = 512
    args["num_channels"] = 32
    args["class_cond"] = False
    args["num_res_blocks"] = 2
    args["num_heads"] = 8
    args["learn_sigma"] = True
    args["use_scale_shift_norm"] = False
    args["attention_resolutions"] = "16,32"
    args["channel_mult"] = "1,2,4,8,8,8"

    # DIFFUSION
    diffusion_args["diffusion_steps"] = 1000
    diffusion_args["noise_schedule"] = "linear"
    diffusion_args["rescale_learned_sigmas"] = False
    diffusion_args["rescale_timesteps"] = False

    diffusion_args.pop("diffusion_steps")
    diffusion_args.pop("timestep_respacing")

    # TODO: change?
    diffusion_args["learn_sigma"] = True

    args = args | diffusion_args

    model, diffusion = create_model_and_diffusion(**args)

    MODEL = "/exp/sbnd/data/users/gputnam/training-SBND/iterE/results/brats2update111000.pt"

    model.load_state_dict(
        dist_util.load_state_dict(MODEL, map_location="cpu")
    )

    model.to(dist_util.dev())
    _ = model.eval()


    save_data = {}

    subfiles = os.listdir(input_base_dir)
    filenames = [f for f in subfiles if f.startswith("g4-raw-")]
    for filename in tqdm(filenames):
        print(f"Processing {filename}")
        _cache_file, _cache_true = load_data(input_base_dir + "/" + filename, threshold=threshold)

        for bidx, bstart in tqdm(enumerate(range(0, len(_cache_file), batch_size))):
            print(f"Processing batch {bidx}")
            imgs = th.tensor(_cache_file[bstart:bstart+batch_size], device=dist_util.dev())

            for iimg in range(len(imgs)):
                save_data[bstart + iimg] = {
                    "original": imgs[iimg].cpu().numpy()
                }

            for T in range(100, 500, 100):
                ddim_noised, rand_noised = noise_frames(diffusion, model, imgs, T)
                recos = reconstruct_frames(diffusion, model, imgs, ddim_noised, rand_noised, T)

                for iimg in range(len(imgs)):
                    iidx = bstart + iimg
                    save_data[iidx]["ddim_noised-T%d"%T] = ddim_noised[iimg].cpu().numpy()
                    save_data[iidx]["rand_noised-T%d"%T] = rand_noised[iimg].cpu().numpy()
                    save_data[iidx]["ddim2ddim-T%d"%T] = recos[0][iimg].cpu().numpy()
                    save_data[iidx]["rand2ddim-T%d"%T] = recos[1][iimg].cpu().numpy()
                    save_data[iidx]["ddim2ddpm-T%d"%T] = recos[2][iimg].cpu().numpy()
                    save_data[iidx]["rand2ddpm-T%d"%T] = recos[3][iimg].cpu().numpy()

        with open(save_dir + "/"+filename.split(".")[0]+"-results.pkl", "wb") as f:
            pickle.dump(save_data, f)
        print(f"Saved results for {filename}")