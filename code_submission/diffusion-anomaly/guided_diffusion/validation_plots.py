import torch as th 
import numpy as np
import matplotlib.pyplot as plt
from . import dist_util

def validation_plots(diffusion, model, basedir, img0, ddpm=True):
    genf = diffusion.p_sample_loop_progressive if ddpm else diffusion.ddim_sample_loop_progressive
    i0 = img0.cpu().numpy()
    img0 = img0.to(dist_util.dev())

    idiffs = []
    ts = []

    plt.imshow(np.squeeze(i0), vmin=-1, vmax=1)
    plt.colorbar()
    plt.savefig(basedir + "img.png", bbox_inches="tight")
    plt.title("Original Image")
    plt.close()

    for i, t in enumerate(range(0, 1100, 100)):
        if i == 0: continue

        t = th.tensor(t-1, device=dist_util.dev())
        idiff = diffusion.q_sample(img0, t)
        idiffs.append(idiff)
        ts.append(t)

    for i, (idiff, t) in enumerate(zip(idiffs, ts)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
        c1 = ax1.imshow(np.squeeze(idiff.cpu().numpy()))
        fig.colorbar(c1, ax=ax1)

        c2 = ax2.imshow(np.squeeze(idiff.cpu().numpy()) - np.squeeze(i0))
        fig.colorbar(c2, ax=ax2)

        plt.title("Diffused T: %i" % int(t))
        plt.savefig(basedir + "diffusion_t%i.png" % int(t), bbox_inches="tight")
        plt.close()

    gens = []
    for (t, idiff) in zip(ts, idiffs):
        gen = genf(model, i0.shape, time=t.item(), noise=idiff.unsqueeze(0))
        final = None
        for g in gen:
            final = g
        gens.append(final)

    for i, (idiff, g, t) in enumerate(zip(idiffs, gens, ts)):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16,4))
        c1 = ax1.imshow(np.squeeze(idiff.cpu().numpy()), vmin=-1, vmax=1)
        fig.colorbar(c1, ax=ax1)
        ax1.title.set_text("Diffused")
    
        c2 = ax2.imshow(np.squeeze(g["sample"].cpu().numpy()), vmin=-1, vmax=1)
        fig.colorbar(c2, ax=ax2)
        ax2.title.set_text("Reconstructed")

        c3 = ax3.imshow(np.squeeze(g["sample"].cpu().numpy()) - np.squeeze(i0))
        fig.colorbar(c3, ax=ax3)
        ax3.title.set_text("Reconstructed - Original")
    
        c4 = ax4.imshow(np.squeeze(g["sample"].cpu().numpy()) - np.squeeze(idiff.cpu().numpy()))
        fig.colorbar(c4, ax=ax4)
        ax4.title.set_text("Reconstructed - Diffused")
        plt.savefig(basedir + "reconstructed_t%i.png" % int(t), bbox_inches="tight")
        plt.close()

    for tind in range(0, i0.shape[1], 100):
        for t, g in list(zip(ts, gens))[::2]:
            lbl = "DDPM" if ddpm else "DDIM"
            plt.plot(np.squeeze(g["sample"].cpu().numpy())[:, tind], label=f"{lbl}, T={t}")
        plt.plot(np.squeeze(i0)[:, tind], label="Original", linewidth=3, color="black", linestyle="--")
        plt.legend()

        plt.title("Reconstructed Waveform, I: %i" % tind)
        plt.savefig(basedir + "reco_wavf_I%i.png" % tind, bbox_inches="tight")
        plt.close()

    plt.plot(np.abs(np.fft.rfft(np.squeeze(i0), axis=0)).sum(axis=1)[1:], label="Original")
    for t, g in list(zip(ts, gens))[::2]:
        plt.plot(np.abs(np.fft.rfft(np.squeeze(g["pred_xstart"].cpu().numpy()), axis=0)).sum(axis=1)[1:], label=f"DDIM, T={t}")

    plt.legend()
    plt.title("Time-Direction Noise Spectrum")
    plt.savefig(basedir + "time_ffts.png", bbox_inches="tight")
    plt.close()

    plt.plot(np.abs(np.fft.rfft(np.squeeze(i0), axis=1)).sum(axis=0)[1:], label="Original")
    for t, g in list(zip(ts, gens))[::2]:
        plt.plot(np.abs(np.fft.rfft(np.squeeze(g["pred_xstart"].cpu().numpy()), axis=1)).sum(axis=0)[1:], label=f"DDIM, T={t}")

    plt.legend()
    plt.title("Wire-Direction Noise Spectrum")
    plt.savefig(basedir + "wire_ffts.png", bbox_inches="tight")
    plt.close()

    loop = list(genf(model, i0.shape, time=ts[-1].item(), noise=idiffs[-1].unsqueeze(0)))

    for i, img in enumerate(loop[99::100]):
        plt.figure()
        plt.imshow(np.squeeze(img["sample"].cpu().numpy()), vmin=-1, vmax=1)
        plt.colorbar()
        plt.title("Reconstruction Series, Step: %i" % (i*100+99))
        plt.savefig(basedir + "reco_img_t%i_to_%i.png" % (int(ts[-1]), i*100+99), bbox_inches="tight")
        plt.close()

    for tind in range(0, i0.shape[1], 100):
        arr = [np.squeeze(img["sample"].cpu().numpy())[:, tind] for img in loop]
        arr = np.array(arr)
        plt.imshow(arr.T, vmin=-1, vmax=1)
        plt.colorbar()
        plt.title("Reconstruction Series, I: %i" % (tind))
        plt.savefig(basedir + "reco_loop_I%i.png" % tind, bbox_inches="tight")
        plt.close()
