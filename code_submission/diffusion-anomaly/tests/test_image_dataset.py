import numpy as np

from guided_diffusion.image_datasets import ImageDataset


def test_image_dataset_yields_npz_array_and_dict(tmp_npz):
    npz_path, n, h, w = tmp_npz

    ds = ImageDataset(
        resolution=h,
        image_paths=str(npz_path.parent),
        classes=None,
        charge_scale=1.0,
        require_charge=False,
        importance_sampling=False,
    )
    it = iter(ds)
    arr, out_dict = next(it)

    # ImageDataset.__iter__ yields a single (1, H, W) slice from the cached array
    assert arr.shape == (1, h, w)
    assert arr.dtype == np.float32
    assert "path" in out_dict
    assert "weight" in out_dict
    assert "pixel_weight" in out_dict
    assert out_dict["path"] == "sample"
