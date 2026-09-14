# Convert waveform .root file to h5 sample file

import argparse
import gc
import os
from os import path

import h5py
import uproot
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--n_events_per_file", type=int, default=10)
parser.add_argument("--tpc", type=int, default=0)
parser.add_argument("--this_dir", type=str, required=True)
parser.add_argument("--save_path", type=str, default=None)
parser.add_argument("--chunk_size", type=int, default=25,
                    help="Reopen ROOT file every N events to release uproot caches")

args = parser.parse_args()

WIRE_EDGES = {
    0: [0, 1984, 3968, 5638],  # tpc0
    1: [5638, 5638 + 1984, 5638 + 3968, 5638 + 5638],  # tpc1
}


if __name__ == "__main__":
    this_dir = args.this_dir
    save_path = args.save_path if args.save_path is not None else this_dir
    os.makedirs(save_path, exist_ok=True)
    tpc = args.tpc
    n_events_per_file = args.n_events_per_file
    chunk_size = args.chunk_size
    tag_raw = "raw"
    wire_lo, wire_hi = WIRE_EDGES[tpc][0], WIRE_EDGES[tpc][-1]

    this_raw_file = path.join(this_dir, "waveform.root")
    tree_path = this_raw_file + ":wvfm/raw_wvfm"

    with uproot.open(tree_path) as wvfm:
        all_keys = list(wvfm.keys())
    n_events = len(all_keys)

    nfile = 0
    wnevts = 0
    save_name_tag_raw = "g4-raw-{}_{}".format(tpc, nfile)
    hf_raw = h5py.File(path.join(save_path, "{}.h5".format(save_name_tag_raw)), "w")

    pbar = tqdm(total=n_events, desc="events")
    for chunk_start in range(0, n_events, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_events)
        with uproot.open(tree_path) as wvfm:
            for evtno in range(chunk_start, chunk_end):
                arr = wvfm[all_keys[evtno]].values()
                this_arr = arr[wire_lo:wire_hi]
                del arr

                g1 = hf_raw.create_group("/%d" % wnevts)
                g1.create_dataset(tag_raw, data=this_arr)
                del this_arr
                wnevts += 1
                pbar.update(1)

                if wnevts == n_events_per_file:
                    print("moving to next file")
                    hf_raw.flush()
                    hf_raw.close()
                    nfile += 1
                    save_name_tag_raw = "g4-raw-{}_{}".format(tpc, nfile)
                    hf_raw = h5py.File(path.join(save_path, "{}.h5".format(save_name_tag_raw)), "w")
                    wnevts = 0
        gc.collect()

    pbar.close()

    if wnevts > 0:
        hf_raw.flush()
        hf_raw.close()
    elif wnevts == 0 and nfile > 0:
        hf_raw.close()
        os.remove(path.join(save_path, "g4-raw-{}_{}.h5".format(tpc, nfile)))

    gc.collect()
    print("done")
