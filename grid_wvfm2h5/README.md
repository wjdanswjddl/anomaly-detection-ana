# Grid-integrated waveform → HDF5 conversion

## Decision

**Chosen: C++ converter in the LArSoft `endscript` (option 1).**

| | C++ endscript | Python `jobsub` (cafpyana-style) |
|---|---|---|
| Runs as part of waveform production | Yes | No (second stage) |
| Re-copies multi-GB `waveform.root` | No | Yes (`xrdcp`) |
| Fits existing `project.py` XML | Yes | Separate submitter |
| Memory / deps | ROOT + UPS `hdf5` on worker | Needs packaged Python/h5py/uproot |
| Matches prior offline `wvfm2h5.py` layout | Yes (`g4-raw-{tpc}_{n}.h5`) | Yes |

Python/cafpyana remains useful for one-off conversion of *existing* pnfs trees, but not for embedding in production jobs.

## What was added

- `wvfm2h5.cpp` — ROOT+HDF5 C++ converter (TPC0 and TPC1)
- `wvfm2h5_run.sh` — compile-on-worker + run wrapper
- `wirecell_macro.sh` — endscript: `dump_waveform_all.C` then `wvfm2h5_run.sh`
- `pack_tarball.sh` — rebuilds `/pnfs/sbnd/resilient/users/munjung/dump_waveform.tar`
- `raw_samples_wvfm2h5_test.xml` — 2-job smoke test (`datafiletypes=root,h5`)

## Submit a small test

```bash
source /nashome/m/munjung/setup_EL9.sh
source /nashome/m/munjung/setup_dnnsp.sh   # if larsoft env needed for local tools

cd /exp/sbnd/app/users/munjung/anomaly-detection/grid_wvfm2h5
bash pack_tarball.sh

# after larbatch/project.py is on PATH (setup_EL9 does this):
project.py --xml raw_samples_wvfm2h5_test.xml --stage celltree --submit
project.py --xml raw_samples_wvfm2h5_test.xml --stage celltree --checkana
```

Outputs land in:
`/pnfs/sbnd/scratch/users/munjung/v10_06_00/wvfm2h5_grid_test/waveforms/<jobid>/`
Expect `waveform.root`, `celltree.root`, and `g4-raw-0_*.h5` / `g4-raw-1_*.h5`.

## Production XML note

For full campaigns, set in the stage:

```xml
<datafiletypes>root,h5</datafiletypes>
<endscript>.../wirecell_macro.sh</endscript>
```

and rebuild the dropbox tarball with `pack_tarball.sh` after any converter change.
