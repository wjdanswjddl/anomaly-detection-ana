// wvfm2h5.cpp
// Convert waveform.root (TH2F under wvfm/raw_wvfm) to g4-raw-{tpc}_{n}.h5
// Layout matches anomaly-detection/wvfm2h5.py
//
// Uses HDF5 C API (no H5Cpp) for portability across UPS/spack builds.
// Streams one TH2 at a time (do not load all events into memory).

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "TFile.h"
#include "TH2.h"
#include "TH2F.h"
#include "TKey.h"
#include "TList.h"
#include "TSystem.h"

#include "hdf5.h"

namespace {

constexpr int kWireEdges[2][2] = {
    {0, 5638},
    {5638, 5638 + 5638},
};

void write_dataset(hid_t grp, const char* name, const std::vector<float>& data,
                   hsize_t nrow, hsize_t ncol) {
  hsize_t dims[2] = {nrow, ncol};
  hid_t space = H5Screate_simple(2, dims, nullptr);
  hid_t ds = H5Dcreate2(grp, name, H5T_NATIVE_FLOAT, space, H5P_DEFAULT,
                        H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  H5Dclose(ds);
  H5Sclose(space);
}

struct TpcWriter {
  int tpc;
  std::string outdir;
  int n_events_per_file;
  int nfile = 0;
  int wnevts = 0;
  hid_t hf = H5I_INVALID_HID;
  char path[512]{};

  void open_next() {
    if (hf >= 0) {
      H5Fclose(hf);
      hf = H5I_INVALID_HID;
    }
    std::snprintf(path, sizeof(path), "%s/g4-raw-%d_%d.h5", outdir.c_str(), tpc, nfile);
    hf = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (hf < 0) {
      throw std::runtime_error(std::string("failed to create ") + path);
    }
    std::cout << "opened " << path << std::endl;
  }

  void write_event(TH2* h) {
    auto* h2f = dynamic_cast<TH2F*>(h);
    if (!h2f) {
      throw std::runtime_error("expected TH2F in raw_wvfm");
    }
    const int wire_lo = kWireEdges[tpc][0];
    const int wire_hi = kWireEdges[tpc][1];
    const int n_wires = wire_hi - wire_lo;
    const int nx = h2f->GetNbinsX();
    const int ny = h2f->GetNbinsY();
    if (wire_hi > nx) {
      std::cerr << "WARNING: hist nx=" << nx << " < wire_hi=" << wire_hi << std::endl;
    }

    if (hf < 0) open_next();

    // TH2F array includes under/overflow: index = iy*(nx+2) + ix
    const Float_t* src = h2f->GetArray();
    const int stride = nx + 2;
    std::vector<float> arr(static_cast<size_t>(n_wires) * ny, 0.f);
    const int xmax = std::min(wire_hi, nx);
    for (int iw = wire_lo; iw < xmax; ++iw) {
      const int row = iw - wire_lo;
      const int ix = iw + 1;  // ROOT bin
      for (int it = 0; it < ny; ++it) {
        arr[static_cast<size_t>(row) * ny + it] = src[(it + 1) * stride + ix];
      }
    }

    char gname[32];
    std::snprintf(gname, sizeof(gname), "/%d", wnevts);
    hid_t grp = H5Gcreate2(hf, gname, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    write_dataset(grp, "raw", arr, static_cast<hsize_t>(n_wires),
                  static_cast<hsize_t>(ny));
    H5Gclose(grp);
    ++wnevts;

    if (wnevts == n_events_per_file) {
      std::cout << "moving to next file (tpc=" << tpc << ")" << std::endl;
      ++nfile;
      wnevts = 0;
      open_next();
    }
  }

  void finish() {
    if (hf < 0) return;
    if (wnevts == 0 && nfile > 0) {
      std::snprintf(path, sizeof(path), "%s/g4-raw-%d_%d.h5", outdir.c_str(), tpc, nfile);
      H5Fclose(hf);
      hf = H5I_INVALID_HID;
      gSystem->Unlink(path);
      std::cout << "removed empty " << path << std::endl;
    } else {
      H5Fclose(hf);
      hf = H5I_INVALID_HID;
    }
  }
};

}  // namespace

int main(int argc, char** argv) {
  std::string input = "waveform.root";
  std::string outdir = ".";
  int n_events_per_file = 10;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if ((a == "--input" || a == "-i") && i + 1 < argc) {
      input = argv[++i];
    } else if ((a == "--outdir" || a == "-o") && i + 1 < argc) {
      outdir = argv[++i];
    } else if ((a == "--n_events_per_file" || a == "-n") && i + 1 < argc) {
      n_events_per_file = std::atoi(argv[++i]);
    } else if (a == "--help" || a == "-h") {
      std::cout << "Usage: " << argv[0]
                << " [--input waveform.root] [--outdir .] [--n_events_per_file 10]\n";
      return 0;
    }
  }

  gSystem->mkdir(outdir.c_str(), true);

  TFile* fin = TFile::Open(input.c_str(), "READ");
  if (!fin || fin->IsZombie()) {
    std::cerr << "ERROR: cannot open " << input << std::endl;
    return 1;
  }

  TDirectory* rawdir = dynamic_cast<TDirectory*>(fin->Get("wvfm/raw_wvfm"));
  if (!rawdir) {
    std::cerr << "ERROR: missing wvfm/raw_wvfm in " << input << std::endl;
    fin->Close();
    return 1;
  }

  TList* keys = rawdir->GetListOfKeys();
  if (!keys || keys->GetSize() == 0) {
    std::cout << "No events in raw_wvfm; nothing to convert." << std::endl;
    fin->Close();
    return 0;
  }

  TpcWriter w0{0, outdir, n_events_per_file};
  TpcWriter w1{1, outdir, n_events_per_file};

  int n_loaded = 0;
  try {
    for (int i = 0; i < keys->GetSize(); ++i) {
      TKey* key = dynamic_cast<TKey*>(keys->At(i));
      if (!key) continue;
      TObject* obj = key->ReadObj();
      TH2* h = dynamic_cast<TH2*>(obj);
      if (!h) {
        delete obj;
        continue;
      }
      w0.write_event(h);
      w1.write_event(h);
      delete h;
      ++n_loaded;
      if (n_loaded % 5 == 0) {
        std::cout << "converted " << n_loaded << " events..." << std::endl;
      }
    }
    w0.finish();
    w1.finish();
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    fin->Close();
    return 2;
  }

  std::cout << "Loaded/converted " << n_loaded << " events from " << input << std::endl;
  fin->Close();
  std::cout << "done" << std::endl;
  return 0;
}
