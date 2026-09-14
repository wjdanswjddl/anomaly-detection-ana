# Convert waveform .root file to h5 sample file

import uproot as uproot
import numpy as np
import pandas as pd
from tqdm import tqdm

import h5py
import os
from os import path
import matplotlib.pyplot as plt

WIRE_EDGES = {0: [0, 1984, 3968, 5638,], # tpc0
              1: [5638, 5638+1984, 5638+3968, 5638+5638]} # tpc1


with open("/home/munjung/anomaly-detection/wvfms_xrootd.list", "r") as f:
    raw_lines = [line.strip() for line in f if line.strip()]
print(f"number of raw files: {len(raw_lines)}")

save_path = "/scratch/7DayLifetime/munjung/anomaly-detection"
sudir_tag = "bnb_cosmics"
print("output directory: ", save_path)

# save configs
tag_raw = "raw"
tpc = 0
save_name_tag_raw = f"g4-raw-{tpc}.h5"

n_events_per_file = 10
print(f"saving h5 files with raw frames of tpc {tpc} with {n_events_per_file} events per file")


if __name__ == "__main__":
    nfile = 0
    wnevts = 0

    subdir = f"{sudir_tag}_{nfile}"
    if not os.path.exists(path.join(save_path, subdir)):
        os.makedirs(path.join(save_path, subdir))
    hf_raw = h5py.File(path.join(save_path, subdir, f"{save_name_tag_raw}"), "w") 

    for i in tqdm(range(len(raw_lines))):
        this_raw_file = raw_lines[i]
        wvfm = uproot.open(this_raw_file+":wvfm/raw_wvfm")
        print("number of events in wvfm", len(wvfm.keys()))
        klist = list(wvfm.keys())

        for evtno in range(len(klist)):
            arr = wvfm[klist[evtno]].values()
            this_arr = arr[WIRE_EDGES[tpc][0]:WIRE_EDGES[tpc][-1]]
            g1 = hf_raw.create_group('/%d'%(wnevts))
            g1.create_dataset(tag_raw, data=this_arr)
            wnevts = wnevts + 1

            if wnevts == n_events_per_file:
                print("moving to next file")
                hf_raw.close()
                nfile  = nfile + 1
                subdir = f"{sudir_tag}_{nfile}"
                if not os.path.exists(path.join(save_path, subdir)):
                    os.makedirs(path.join(save_path, subdir))
                hf_raw = h5py.File(path.join(save_path, subdir, f"{save_name_tag_raw}"), "w") 
                wnevts = 0

        wvfm.close()

    print("done")