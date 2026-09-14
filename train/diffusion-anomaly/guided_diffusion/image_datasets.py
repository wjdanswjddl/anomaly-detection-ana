import math
import random
from pathlib import Path
from PIL import Image
import blobfile as bf
import numpy as np
import torch as th
from torch.utils.data import DataLoader, IterableDataset, Dataset, BatchSampler, RandomSampler, SequentialSampler
from torchvision import transforms
from .train_util import visualize
from visdom import Visdom
# viz = Visdom(port=8850)
viz = Visdom(port=8850, server="sbndbuild03.fnal.gov")
from scipy import ndimage


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=True,
    random_crop=False,
    random_flip=False,
    require_charge=False,
    importance_sampling=False,
    importance_maxwgt=10,
    charge_scale=1,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)

    classes = None

    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.

        class_names =[path.split("/")[6] for path in all_files] #9 or 3
        print('classnames', class_names)


        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]

    dataset = ImageDataset(
        image_size,
        data_dir,
        classes=classes,
        shard=0,
        num_shards=1,
        random_crop=random_crop,
        random_flip=random_flip,
        importance_sampling=importance_sampling,
        importance_maxwgt=importance_maxwgt,
        charge_scale=charge_scale,
        require_charge=require_charge,
    )

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )

    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "npy", "npz"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(IterableDataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        importance_sampling=False,
        importance_maxwgt=10.,
        charge_scale=1.,
        require_charge=False,
        random_crop=False,
        random_flip=False,
        exts=['jpg', 'jpeg', 'png', 'npy', 'npz']
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = [p for ext in exts for p in Path(f'{image_paths}').glob(f'**/*.{ext}')]

        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

        self.importance_sampling = importance_sampling
        self.importance_maxwgt = importance_maxwgt
        self.charge_scale = charge_scale
        self.require_charge = require_charge

        self.idx = -1
        self._cache_file = None
        self._cache_find = -1
        self._cache_aind = -1
        self._cache_fname = None
        self._weights = None

    def _getnext(self):
        self.idx += 1
        self._cache_aind += 1

        if self._cache_file is None or self._cache_aind >= self._cache_file.shape[0]:
            self._cache_find += 1
            self._cache_aind = 0

            if self._cache_find >= len(self.local_images):
                raise StopIteration 

            path = self.local_images[self._cache_find]
            name=str(path).split("/")[-1].split(".")[0]
            numpy_img = np.load(path)
            self._cache_file = visualize(numpy_img["reco"]).astype(np.float32)
            self._cache_fname = name
            self._cache_true = numpy_img["truth"].astype(np.float32)

            # weight by sum of true charge
            self._weights = np.sum(self._cache_true, axis=(1, 2, 3)).astype(np.float32)
            # Normalize average weight to 1
            self._weights = self._weights / np.mean(self._weights)

            # Normalize the charge so that signal and noise pixels are, in total, weighted about the same
            charge_norm = np.mean(self._cache_true)
            # per-pixel map of 1 + normalized true charge 
            self._pixel_weights = 1 + self._cache_true / charge_norm / self.charge_scale
            # normalized to one
            self._pixel_weights = self._pixel_weights / np.mean(self._pixel_weights)
            self._charge = np.sum(self._cache_true, axis=(1, 2, 3)).astype(np.float32)

        arr = self._cache_file[self._cache_aind]
        w = self._weights[self._cache_aind]
        pw = self._pixel_weights[self._cache_aind]
        c = self._charge[self._cache_aind]

        return arr, w, pw, c 

    def __iter__(self):
        while True:
            while True:
                try:
                    arr, w, pw, c = self._getnext()
                except StopIteration:
                    print("EPOCH COMPLETED. RESTARTING.")
                    self._cache_find = 0
                    continue
                except Exception as e:
                    print("Opening file (%s) failed with error: %s. Skipping..." % (self.local_images[self._cache_find], str(e)))
                    self._cache_find += 1
                    continue

                # ignore events with no charge
                if self.require_charge and c < 1:
                    continue
                if not self.importance_sampling or (w/self.importance_maxwgt) > np.random.rand():
                    break

            # If we are importance weighting, then the weight is now 1, so as to not double-count
            if self.importance_sampling:
                w[:] = 1.

            out_dict = {}
            if self.local_classes is not None:
                out_dict["y"] = np.array(self.local_classes[self._cache_find], dtype=np.int64)

            out_dict["path"] = self._cache_fname
            out_dict["weight"] = w
            out_dict["pixel_weight"] = pw

            yield arr, out_dict


class ShuffleDataset(IterableDataset):
    def __init__(self, dataset, buffer_size=1000):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size
        self._iterator = None  # To store the state

    def __iter__(self):
        # This creates a generator, which IS an iterator
        shufbuf = []
        try:
            dataset_iter = iter(self.dataset)
            for _ in range(self.buffer_size):
                shufbuf.append(next(dataset_iter))
        except StopIteration:
            pass

        try:
            while True:
                item = next(dataset_iter)
                idx = random.randint(0, len(shufbuf) - 1)
                yield shufbuf[idx]
                shufbuf[idx] = item
        except StopIteration:
            random.shuffle(shufbuf)
            for item in shufbuf:
                yield item

    # Adding this often resolves 'is not an iterator' in complex loaders
    def __next__(self):
        if self._iterator is None:
            self._iterator = iter(self)
        return next(self._iterator)


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 3* image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
   # crop_y=64; crop_x=64
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

def zeropatch(pil_image, image_size):
    im=np.array(th.zeros(image_size, image_size,3))
    arr = np.array(pil_image)
    crop_x = (-arr.shape[0] + image_size)
    crop_y = abs(arr.shape[1] - image_size) // 2
  #  print('crop', crop_y, crop_x) #crop_y=64; crop_x=64
    im[0:arr.shape[0] , crop_y : crop_y +arr.shape[1],:]=arr

    return im#arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]



def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
