import sys
sys.path.append("anomaly-detection/diffusion-anomaly")
sys.path.append("anomaly-detection/diffusion-anomaly/scripts")

import argparse
import os
from tqdm import tqdm

from torch.autograd import Variable
from guided_diffusion.bratsloader import BRATSDataset
import blobfile as bf
import torch as th
os.environ['OMP_NUM_THREADS'] = '8'

import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import numpy as np

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data, ShuffleDataset
from guided_diffusion.train_util import visualize
from guided_diffusion.validation_plots import validation_plots
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
    model_and_diffusion_defaults,
    diffusion_defaults,
    create_model_and_diffusion,
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict
from scripts.classifier_train import compute_top_k

import pickle
import h5py


def load_data(filename):
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
            allarrs.append(ll)
            

    _cache_file = visualize(np.expand_dims(np.concatenate(allarrs), axis=1)).astype(np.float32)
    _cache_true = np.ones((_cache_file.shape[0], 1, nrows, ncols)).astype(np.float32)
    return _cache_file, _cache_true


# plotter to plot and save stuff
def validation_plots_wclass(
    diffusion,
    model,
    basedir,
    img0,
    ddpm=True,
    classifier=None,
    classifier_scale=100.0,
    target_class=0,
):
    this_save_data = {}

    genf = diffusion.p_sample_loop_progressive if ddpm else diffusion.ddim_sample_loop_progressive
    i0 = img0.cpu().numpy()
    img0 = img0.to(dist_util.dev())

    model_kwargs = {}

    cond_fn = None
    if classifier is not None:

        def cond_fn(x, t, y=None):
            assert y is not None
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[th.arange(len(logits), device=logits.device), y.view(-1)]
                grad = th.autograd.grad(selected.sum(), x_in)[0]
                return grad, grad * classifier_scale

    idiffs = []
    ts = []

    this_save_data["original"] = np.squeeze(i0)

    tvals = range(50, 400, 50)
    for i, t in enumerate(tvals):
        if i == 0:
            continue

        t = th.tensor(t - 1, device=dist_util.dev())

        if ddpm:
            # gaussian noise
            idiff = diffusion.q_sample(img0, t)

        else:
            # ddim noise
            idiff = diffusion.ddim_sample_loop_progressive(
                model, 
                img0.shape, 
                time=t.item(), 
                noise=img0.unsqueeze(0), 
                reverse=True,
                progress=True)

            final = None
            for g in idiff:
                final = g
            idiff = final["sample"].squeeze(0)

        idiffs.append(idiff)
        ts.append(t)


    gens = []
    for (t, idiff) in zip(ts, idiffs):
        gen = genf(
            model,
            i0.shape,
            time=t.item(),
            noise=idiff.unsqueeze(0),
            progress=True
        )
        final = None
        for g in gen:
            final = g
        gens.append(final)


        step_result = {
            "diffused": idiff.cpu().numpy(),
            "reconstructed": g["sample"].cpu().numpy(),
            "t": t,
            "noise_spectrum_time": np.abs(np.fft.rfft(np.squeeze(g["pred_xstart"].cpu().numpy()), axis=0)).sum(axis=1)[1:],
            "noise_spectrum_wire": np.abs(np.fft.rfft(np.squeeze(g["pred_xstart"].cpu().numpy()), axis=1)).sum(axis=0)[1:],
        }
        if "saliency" in g:
            step_result["saliency"] = g["saliency"].cpu().numpy()
        this_save_data[f"step_{t}"] = step_result

    return this_save_data


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_base_dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--noise_type", type=str, default="ddpm")
    runargs = parser.parse_args()

    input_base_dir = runargs.input_base_dir

    threshold = runargs.threshold
    print("threshhold: ", threshold)

    if runargs.noise_type == "ddpm":
        is_ddpm = True
    else:
        is_ddpm = False
    print("DDPM: ", is_ddpm)

    #load model and diffusion
    args = model_and_diffusion_defaults()
    diffusion_args = diffusion_defaults()

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

    # input configs
    # take input base dir as argument
    # input_base_dir = "/scratch/7DayLifetime/munjung/anomaly-detection-raw-h5/83487027_3"
    subfiles = os.listdir(input_base_dir)
    filenames = [f for f in subfiles if f.startswith("g4-raw-")]

    # save configs
    if threshold > 0.0:
        save_dir = input_base_dir + f"/results_interesting-grayNN-{threshold}-{runargs.noise_type}"
        print("saveing frames with interesting images")
    else:
        save_dir = input_base_dir + f"/results-grayNN-{threshold}-{runargs.noise_type}"
        print("saveing frames with all images")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for filename in filenames:
        save_data = {}
        print(f"Processing {filename}")
        _cache_file, _cache_true = load_data(input_base_dir + "/" + filename)
        print(f"Loaded {len(_cache_file)} images")
        for img_idx in tqdm(range(len(_cache_file))):
            this_img = _cache_file[img_idx]
            # arr_reco[img_idx] is [C,H,W]; add batch dim -> [1,C,H,W]

            # TODO: save only interesting images?
            if threshold > 0.0:
                if np.max(this_img) < threshold:
                    continue

            this_img = th.tensor(this_img, device=dist_util.dev()) #.unsqueeze(0).unsqueeze(0)
            with th.no_grad():
                save_data[img_idx] = validation_plots_wclass(diffusion, model, save_dir, this_img, ddpm=is_ddpm)

            # save every 100 images, just in case
            if img_idx % 50 == 0:
                with open(save_dir + "/"+filename.split(".")[0]+".pkl", "wb") as f:
                    pickle.dump(save_data, f)

        with open(save_dir + "/"+filename.split(".")[0]+".pkl", "wb") as f:
            pickle.dump(save_data, f)
