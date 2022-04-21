from typing import List, Dict, Any
import os
import json

# import nibabel as nib

from PIL import Image
import numpy as np
import h5py
from tqdm import tqdm as tqdm_writer
from key2med.utils.logging import tqdm

from datetime import datetime as dt

import torch

from key2med.utils.transforms import Transform, ResizeTransform
from key2med.utils.helper import hash_dict, get_disk_usage, get_file_size, hash_string
import logging
import coloredlogs

coloredlogs.install(level=logging.INFO)
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Path = str
ImagePath = str
Label = int

# nib.imageglobals.logger.level = 40


class ImageReader:
    """
    Basic class for loading images, transforming and caching
    """

    def __init__(self,
                 base_path: Path,
                 image_paths: List[ImagePath],
                 cache_path: str = None,
                 cache_config: Dict = None,
                 in_memory: bool = False,
                 transform: Transform = None,
                 channels: int = 1):
        self.base_path = base_path
        self.in_memory = in_memory
        self.transform = transform 

        self.images = None
        self.cache_file = None
        self.image_path_to_cache_index: Dict[ImagePath, int] = {}
        self.channels = channels

        if cache_path is not None:
            assert cache_config is not None
        if cache_path is not None and os.path.isfile(cache_path):
            logger.info(f'Reading data from cache file {cache_path}')
            self.check_data_file(cache_path)
            self.cache_file = h5py.File(cache_path, 'r')
            cached_image_paths = set([path.decode('utf-8') for path in self.cache_file['image_paths']])
            paths_to_add = [path for path in image_paths if path not in cached_image_paths]
            if len(paths_to_add) > 0:
                logger.info(f'Found {len(paths_to_add)} images not yet in cache. Adding them now.')
                self.cache_file.close()
                self.cache_data(cache_path, paths_to_add, cache_config)
                self.cache_file = h5py.File(cache_path, 'r')
            cached_image_paths = [path.decode('utf-8') for path in self.cache_file['image_paths']]
            self.image_path_to_cache_index = {path: index for index, path in enumerate(cached_image_paths)}

        if self.in_memory:
            logger.info(f'Reading all data into memory.')
            self.images: Dict[ImagePath: Any] = {path: self.load_image(path) for path in tqdm(image_paths, desc='Loading all data into memory')}
            if self.cache_file is not None:
                logger.info(f'Closing cache file {cache_path} after loading all data into memory.')
                self.cache_file.close()
                self.cache_file = None

        if cache_path is not None and not os.path.isfile(cache_path):
            self.create_cache_file(cache_path, image_paths)

    def create_cache_file(self, cache_path, image_paths):
        logger.info(f'No cache file found. Writing all data to cache.')
        dir_path, total, used, free = get_disk_usage(cache_path)
        logger.info(f'Current disk usage in {dir_path}:\n'
                    f'Total: {total}GB\n'
                    f'Used: {used}GB\n'
                    f'Free: {free}GB')
        self.cache_data(cache_path, image_paths)
        dir_path, total, used, free = get_disk_usage(cache_path)
        logger.info(f'Written {get_file_size(cache_path)}GB to {cache_path}.\n'
                    f'Current disk usage in {dir_path}:\n'
                    f'Total: {total}GB\n'
                    f'Used: {used}GB\n'
                    f'Free: {free}GB')

    @property
    def default_transform(self) -> Transform:
        return ResizeTransform(image_dim=224)

    @property
    def image_dim(self):
        return self.transform.image_dim

    def load_image(self, path) -> torch.tensor:
        if self.images is not None:
            image = self.images[path]
            return torch.tensor(image)
        if self.cache_file is not None and path in self.image_path_to_cache_index:
            cache_index = self.image_path_to_cache_index[path]
            image = self.cache_file['data'][cache_index]
            return torch.tensor(image)
        full_image_path = os.path.join(self.base_path, path)
        return self.read_image_file(full_image_path, self.transform, self.channels)

    def read_image_file(self, image_path, transform: Transform) -> torch.tensor:
        raise NotImplementedError

    @staticmethod
    def check_data_file(cache_path: str) -> None:
        """
        Reads one datapoint from the cache file and prints the data shape,
        to make sure the cache file is not empty and the shape of the data
        is correct.
        :param cache_path:
        :return:  None
        :raises: IndexError, KeyError If the cache file is empty or there are other
                 issues.
        """
        file = h5py.File(cache_path, 'r')
        try:
            image_paths = file['image_paths'][0]
            item = file['data'][0]
        except (IndexError, KeyError) as e:
            logger.info(f'No cached data found in file {cache_path}')
            raise e
        logger.info(f'Found data of size {item.shape} in file {cache_path}')

    def cache_data(self, cache_path, image_paths: List[ImagePath], cache_config: Dict = None, batch_size: int = 1000):
        logger.info(f'Starting cache process for file {cache_path}.')
        logger.info(f'Size of dataset: {len(image_paths)} images.')
        logger.info(f'Writing {len(image_paths) // batch_size + 1} batches of {batch_size} images each.')

        file = h5py.File(cache_path, 'a')
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_indices = range(i, min(len(image_paths), i + batch_size))
            batch_paths = [image_paths[i] for i in batch_indices]
            images = [self.load_image(path) for path in batch_paths]
            images = torch.stack(images)
            batch_paths = [path.encode("ascii", "ignore") for path in batch_paths]

            if 'data' not in file:
                file.create_dataset('data', data=images, chunks=True, maxshape=(None,) + images.shape[1:])
                file.create_dataset('image_paths', dtype=h5py.special_dtype(vlen=str), data=batch_paths, chunks=True, maxshape=(None,))
            else:
                file['data'].resize(file['data'].shape[0] + images.shape[0], axis=0)
                file['data'][-images.shape[0]:] = images
                file['image_paths'].resize(file['image_paths'].shape[0] + len(batch_paths), axis=0)
                file['image_paths'][-len(batch_paths):] = batch_paths
            tqdm_writer.write(
                f'Wrote {len(batch_indices)} items to file. Size of saved dataset: {file["data"].shape}')
        file.close()

        if cache_config is not None:
            cache_config_path = cache_path + '.json'
            if not os.path.exists(cache_config_path):
                with open(cache_path + '.json', 'w') as f:
                    f.write(json.dumps(cache_config))
                logger.info(f"Wrote cache config to {cache_path + '.json'}")


class PNGJPGImageReader(ImageReader):

    def read_image_file(self, image_path, transform: Transform, channels) -> torch.tensor:
        """
        Open and transform an image from a given path.
        :param image_path: Path to image.jpg or image.png
        :return: Array with image.
        """
        image = Image.open(os.path.join(self.base_path, image_path))
        image = np.array(image)
        if len(image.shape) == 3:
            image = image[:, :, 0]
        if transform is not None:
            image = transform(image)
        if image.shape[0] != channels:
            image = image[:channels,:,:]
        return torch.tensor(image)


class NIIImageReader(ImageReader):

    def monochrome_value_from_filename(self, image_path):
        m = os.path.basename(image_path).split('_')[3]
        return m

    def read_image_file(self, image_path, transform: Transform) -> torch.tensor:
        monochrome_value = self.monochrome_value_from_filename(image_path)
        image: np.array = nib.load(image_path).get_fdata().T
        if monochrome_value == 'M1':
            # M1 monochrome has max value 1024 and is inverted
            image /= 1024.
            image = 1 - image
        elif monochrome_value == 'M2':
            # M2 monochrome has max value 4096
            image /= 4096.
        else:
            raise NotImplementedError(f'monochrome value "{monochrome_value}" not supported')
        # convert image to [0, 255] because all transform assume this range
        image *= 255.
        if transform is not None:
            image = transform(image)
        return torch.tensor(image)


def main():
    base_path = "/data/MEDICAL/datasets/CheXpert_small/CheXpert-v1.0-small"

    all_images = [
    'train/patient00001/study1/view1_frontal.jpg',
    'train/patient00002/study2/view1_frontal.jpg',
    'train/patient00002/study1/view1_frontal.jpg',
    'train/patient00002/study1/view2_lateral.jpg',
    'train/patient00003/study1/view1_frontal.jpg',
    'train/patient00004/study1/view1_frontal.jpg',
    'train/patient00004/study1/view2_lateral.jpg',
    'train/patient00005/study1/view1_frontal.jpg',
    'train/patient00005/study1/view2_lateral.jpg',
    'train/patient00005/study2/view1_frontal.jpg',
    'train/patient00005/study2/view2_frontal.jpg',
    'train/patient00006/study1/view1_frontal.jpg',
    'train/patient00007/study1/view1_frontal.jpg',
    'train/patient00007/study2/view1_frontal.jpg',
    'train/patient00008/study1/view1_frontal.jpg',
    'train/patient00008/study2/view1_frontal.jpg',
    'train/patient00009/study1/view1_frontal.jpg',
    'train/patient00009/study1/view2_lateral.jpg',
    'train/patient00010/study1/view1_frontal.jpg',
    ]
    first_batch = all_images[:10]

    os.makedirs('/tmp/test_caches/', exist_ok=True)
    random_cache_path = os.path.join('/tmp/test_caches', hash_string(str(dt.now())))

    transform = ResizeTransform(image_dim=224)

    image_reader = PNGJPGImageReader(base_path=base_path, image_paths=first_batch, cache_path=random_cache_path, in_memory=False, transform=transform)
    image_reader = PNGJPGImageReader(base_path=base_path, image_paths=all_images, cache_path=random_cache_path, in_memory=True, transform=transform)

    for path in all_images:
        image = image_reader.load_image(path)
        print(image.shape)


if __name__ == '__main__':
    main()
