from typing import List, Tuple, Dict, Optional, Union, Any
import os
import random
from pathlib import Path

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

from key2med.data.image_reader import ImageReader, PNGJPGImageReader
from key2med.data.label_reader import ChexpertLabelReader, PatientLabelReader, MonitoringLabelReader
from key2med.utils.transforms import BaseTransform, Transform, RandomAffineTransform
from key2med.utils.plotting import text_histogram
from key2med.utils.helper import hash_dict

import logging
import coloredlogs

coloredlogs.install(level=logging.INFO)
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ImagePath = str
Label = str

DatasetIndex = int
ImageIndex = int


class BaseDataset(Dataset):
    image_path_mapping: List[Any] = None
    image_data: Dict[ImagePath, Dict] = None
    image_reader: ImageReader = None
    random_transform: Transform = None

    def __getitem__(self, index) -> Tuple[List[torch.tensor], List[Any]]:
        if index >= len(self):
            raise IndexError
        image_path = self.image_path_mapping[index]
        image = self.image_reader.load_image(image_path)
        if self.random_transform is not None:
            image = self.random_transform(image)
        image_data = self.load_image_data(image_path)
        return image, image_data

    def __len__(self):
        return len(self.image_path_mapping)

    def load_image_data(self, image_path):
        raise NotImplementedError


class CheXpertDataset(BaseDataset):

    def __init__(self,
                 data_path: str,
                 split: str,
                 split_config: str = 'train_valid',
                 transform: Transform = None,
                 random_transform: Transform = None,
                 upsample_label: str = None,
                 downsample_label: str = None,
                 max_size: int = None,
                 print_stats: bool = True,
                 plot_stats: bool = False,
                 # label_reader kwargs
                 label_filter: List[int] = None,
                 uncertain_to_one: Union[str, List[str]] = None,
                 uncertain_to_zero: Union[str, List[str]] = None,
                 uncertain_upper_bound: float = 0.5,
                 uncertain_lower_bound: float = 0.5,
                 one_labels: List[str] = None,
                 valid_views: List[str] = None,
                 valid_sexs: List[str] = None,
                 valid_directions: List[str] = None,
                 min_age: int = None,
                 max_age: int = None,
                 # image_reader kwargs
                 use_cache: bool = False,
                 in_memory: bool = True,
                 # multiprocessing kwargs
                 rank: int = 0,
                 world_size: int = 1,
                 channels: int =3,
                 ):
        self.split = split
        self.split_config = split_config
        parent_directory, chexpert_directory = self.split_data_path(data_path)
        label_file, sub_split = self.get_label_file(split, split_config)
        self.label_reader = self.init_label_reader(label_file=os.path.join(parent_directory, chexpert_directory, label_file),
                                                   label_filter=label_filter,
                                                   uncertain_to_one=uncertain_to_one,
                                                   uncertain_to_zero=uncertain_to_zero,
                                                   uncertain_upper_bound=uncertain_upper_bound,
                                                   uncertain_lower_bound=uncertain_lower_bound,
                                                   one_labels=one_labels,
                                                   valid_views=valid_views,
                                                   valid_sexs=valid_sexs,
                                                   valid_directions=valid_directions,
                                                   min_age=min_age,
                                                   max_age=max_age,
                                                   max_size=max_size)
        self.image_paths = self.get_image_paths(self.label_reader)
        if sub_split is not None:
            self.image_paths = self.split_paths(self.image_paths, sub_split=sub_split, train_size=0.9)
        if max_size is not None:
            self.image_paths = self.image_paths[:max_size]

        transform = transform or self.default_transform
        self.random_transform = random_transform 
        cache_config, cache_path = None, None
        if use_cache:
            cache_config = self.get_cache_config(data_path, label_file, transform)
            os.makedirs(os.path.join(data_path, 'cache'), exist_ok=True)
            cache_path = os.path.join(data_path, 'cache', hash_dict(cache_config))

        self.image_reader = self.init_image_reader(base_path=parent_directory,
                                                   image_paths=self.image_paths,
                                                   transform=transform,
                                                   cache_config=cache_config,
                                                   cache_path=cache_path,
                                                   in_memory=in_memory,
                                                   channels=channels)

        if upsample_label is not None:
            self.image_paths = self.upsample_data(self.image_paths, upsample_label)
        if downsample_label is not None:
            self.image_paths = self.downsample_data(self.image_paths, downsample_label)

        self.world_size = world_size
        self.rank = rank
        if self.world_size > 1:
            gpu_split_size = int(len(self) / self.world_size)
            gpu_split_indices = list(range(self.rank * gpu_split_size,
                                           (self.rank + 1) * gpu_split_size))
            self.image_paths = [self.image_paths[i] for i in gpu_split_indices]

        self.image_path_mapping: List[ImagePath] = self.create_image_path_mapping()
        if print_stats:
            self.print_label_stats()
        if plot_stats:
            self.plot_label_stats()

    def split_data_path(self, data_path: str) -> Tuple[str, str]:
        split = Path(data_path).parts
        return str(Path(*split[:-1])), split[-1]

    def init_label_reader(self, **label_reader_kwargs):
        return ChexpertLabelReader(**label_reader_kwargs)

    def init_image_reader(self, **image_reader_kwargs):
        return PNGJPGImageReader(**image_reader_kwargs)

    def get_image_paths(self, label_reader):
        return label_reader.image_paths

    def __len__(self):
        return len(self.image_paths)

    def load_image_data(self, image_path):
        return self.load_image_labels(image_path)

    def load_image_labels(self, image_path):
        return self.label_reader[image_path]

    def load_image_metadata(self, image_path):
        return self.label_reader.data[image_path]

    def upsample_data(self, image_paths, upsample_label: str):
        positive_items = self.get_positive_items(image_paths, upsample_label)
        n_upsample = len(image_paths) - 2 * len(positive_items)
        new_data = random.choices(positive_items, k=n_upsample)
        image_paths += new_data
        return image_paths

    def get_positive_items(self, image_paths, label):
        label_index = self.label_reader.label_to_index[label]
        positive_values = []
        for item in image_paths:
            if self.label_reader.data[item]['labels'][label_index] == 1.:
                positive_values.append(item)
        return positive_values

    def get_negative_items(self, image_paths, label):
        label_index = self.label_reader.label_to_index[label]
        positive_values = []
        for item in image_paths:
            if self.label_reader.data[item]['labels'][label_index] == 0.:
                positive_values.append(item)
        return positive_values

    def downsample_data(self, image_paths, downsample_label: str):
        positive_items = self.get_positive_items(image_paths, downsample_label)
        negative_items = self.get_negative_items(image_paths, downsample_label)
        negative_items = random.choices(negative_items, k=len(positive_items))
        image_paths = positive_items + negative_items
        return image_paths

    @property
    def imratio(self) -> float:
        all_labels = self.all_labels_flat
        return (all_labels.sum() / len(all_labels)).item()

    @property
    def imratios(self) -> List[float]:
        stats = self.label_stats
        return [stat['positive_count'] for stat in stats.values()]

    @property
    def all_labels_flat(self) -> torch.tensor:
       return self.all_labels.view(-1)

    @property
    def index_to_label(self):
        return self.label_reader.index_to_label

    @property
    def all_labels(self) -> torch.tensor:
        labels: List[torch.tensor] = [self.label_reader[path] for path in self.image_path_mapping]
        labels: torch.tensor = torch.stack(labels)
        return labels

    @property
    def label_stats(self) -> Dict[Label, Dict]:
        label_stats: Dict[Label, Dict] = {}
        all_datas = self.all_labels
        n = len(all_datas)
        positive_counts = (all_datas == 1.).sum(axis=0)
        negative_counts = (all_datas == 0.).sum(axis=0)
        uncertain_counts = ((all_datas != .0) & (all_datas != 1.)).sum(axis=0)
        for label_name, pos, neg, unc in zip(self.index_to_label, positive_counts, negative_counts, uncertain_counts):
            label_stats[label_name] = {
                'positive_count': pos,
                'positive_ratio': pos / n,
                'negative_count': neg,
                'negative_ratio': neg / n,
                'uncertain_count': unc,
                'uncertain_ratio': unc / n,
            }
        return label_stats

    def create_image_path_mapping(self) -> List[ImagePath]:
        return self.image_paths

    def get_cache_config(self, data_path, label_file, transform):
        config = {
            'data_path': data_path,
            'label_file': label_file,
            'transform': transform.config
        }
        return config

    @property
    def all_image_paths(self):
        return

    def print_label_stats(self):
        """
        Plot stats on the distribution of labels and image metadata to the
        command line.
        :return: None
        """
        label_stats = self.label_stats
        imratio_message = f'\n\t{"=" * 10} SPLIT {self.split} {"=" * 10}:\n' \
                          f'\tTotal images in split:  {len(self.image_paths):,}\n' \
                          f'\tTotal items in split:  {len(self.image_path_mapping):,}\n' \
                          f'\tTotal imratio in split: {self.imratio:.1%}.\n'

        max_label_length: int = max([len(label) for label in label_stats.keys()])
        for label, stats in label_stats.items():
            imratio_message += f'\t{label: <{max_label_length + 1}}: ' \
                               f'{stats["positive_count"]:>7,} positive, ' \
                               f'{stats["negative_count"]:>7,} negative, ' \
                               f'{stats["uncertain_count"]:>7,} uncertain, ' \
                               f'{stats["positive_ratio"]:>7.1%} imratio.\n'
        logger.info(imratio_message)

    @property
    def all_metadata(self) -> List[Dict]:
        return [self.load_image_metadata(path) for path in self.image_path_mapping]

    def plot_label_stats(self):
        all_labels = self.all_labels_flat
        try:
            text_histogram(all_labels, title=f'distribution of all labels, split: {self.split}')
        except Exception as e:
            logger.info(f'Can not plot stats on data labels: {str(e)}')

        all_metadata = self.all_metadata
        all_sex_values = [data['sex'] for data in all_metadata]
        try:
            text_histogram(all_sex_values, title=f'distribution of sex values, split: {self.split}')
        except Exception as e:
            logger.info(f'Can not plot stats on sex values: {str(e)}')

        all_age_values = [data['age'] for data in all_metadata]
        try:
            text_histogram(all_age_values, title=f'distribution of age values, split: {self.split}')
        except Exception as e:
            logger.info(f'Can not plot stats on age values: {str(e)}')

        all_front_lateral_values = [data['view'] for data in all_metadata]
        try:
            text_histogram(all_front_lateral_values, title=f'distribution of front_lateral values, split: {self.split}')
        except Exception as e:
            logger.info(f'Can not plot stats on front_lateral values: {str(e)}')

        all_ap_pa_values = [data['direction'] for data in all_metadata]
        try:
            text_histogram(all_ap_pa_values, title=f'distribution of ap_pa values, split: {self.split}')
        except Exception as e:
            logger.info(f'Can not plot stats on ap_pa values: {str(e)}')

    @property
    def default_transform(self) -> Transform:
        return BaseTransform()

    @property
    def default_random_transform(self) -> Transform:
        return RandomAffineTransform()

    @property
    def image_dim(self):
        return self.image_reader.image_dim

    @property
    def label_dim(self):
        return self.label_reader.label_dim

    @staticmethod
    def split_paths(paths, sub_split, train_size: float = 0.9) -> List[ImagePath]:
        train_paths, test_paths = train_test_split(paths, train_size=train_size)
        if sub_split == 'train':
            return train_paths
        if sub_split == 'test':
            return test_paths

    def get_label_file(self, split, split_config):
        if split_config == 'train_valid_test':
            if split == 'train':
                return 'train.csv', 'train'
            if split == 'valid':
                return 'train.csv', 'test'
            if split == 'test':
                return 'valid.csv', None
        elif split_config == 'train_valid':
            if split == 'train':
                return 'train.csv', None
            if split == 'valid':
                return 'valid.csv', None
        raise NotImplementedError


class StudyDataset(CheXpertDataset):

    def __init__(self,
                 data_path: str,
                 split: str,
                 split_config: str = 'train_valid',
                 transform: Transform = None,
                 random_transform: Transform = None,
                 max_size: int = None,
                 print_stats: bool = True,
                 plot_stats: bool = False,
                 upsample_label: str = None,
                 downsample_label: str = None,
                 # label_reader kwargs
                 label_filter: List[int] = None,
                 uncertain_to_one: Union[str, List[str]] = None,
                 uncertain_to_zero: Union[str, List[str]] = None,
                 uncertain_upper_bound: float = 0.5,
                 uncertain_lower_bound: float = 0.5,
                 one_labels: List[str] = None,
                 valid_views: List[str] = None,
                 valid_sexs: List[str] = None,
                 valid_directions: List[str] = None,
                 min_age: int = None,
                 max_age: int = None,
                 # image_reader kwargs
                 use_cache: bool = False,
                 in_memory: bool = True,
                 # multiprocessing kwargs
                 rank: int = 0,
                 world_size: int = 1,
                 ):
        super().__init__(
            data_path=data_path,
            split=split,
            split_config=split_config,
            transform=transform,
            random_transform=random_transform,
            max_size=max_size,
            print_stats=print_stats,
            plot_stats=plot_stats,
            upsample_label=upsample_label,
            downsample_label=downsample_label,
            # label_reader kwargs
            label_filter=label_filter,
            uncertain_to_one=uncertain_to_one,
            uncertain_to_zero=uncertain_to_zero,
            uncertain_upper_bound=uncertain_upper_bound,
            uncertain_lower_bound=uncertain_lower_bound,
            one_labels=one_labels,
            valid_views=valid_views,
            valid_sexs=valid_sexs,
            valid_directions=valid_directions,
            min_age=min_age,
            max_age=max_age,
            # image_reader kwargs
            use_cache=use_cache,
            in_memory=in_memory,
            # multiprocessing kwargs
            rank=rank,
            world_size=world_size)

    def __len__(self):
        return len(self.image_path_mapping)

    def init_label_reader(self, **label_reader_kwargs):
        return PatientLabelReader(**label_reader_kwargs)

    def get_image_paths(self, label_reader: PatientLabelReader):
        return label_reader.image_paths

    def create_image_path_mapping(self) -> List[List[ImagePath]]:
        return [study.images for study in self.label_reader.studies]

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        image_paths = self.image_path_mapping[index]
        images = [self.image_reader.load_image(image_path) for image_path in image_paths]
        if self.random_transform is not None:
            images = [self.random_transform(image) for image in images]
        images = torch.stack(images)
        images_data = [self.load_image_data(image_path) for image_path in image_paths]
        labels = images_data[0][0]
        views = torch.stack([data[1] for data in images_data])
        directions = torch.stack([data[2] for data in images_data])
        return images, labels, views, directions

    def load_image_data(self, image_path) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        return self.label_reader[image_path]

    @property
    def all_labels(self):
        all_datas: List[torch.tensor] = [self.label_reader[paths[0]][0] for paths in self.image_path_mapping]
        all_datas: torch.tensor = torch.stack(all_datas)
        return all_datas

    @property
    def all_metadata(self) -> List[Dict]:
        all_paths = [path for paths in self.image_path_mapping for path in paths]
        return [self.load_image_metadata(path) for path in all_paths]


class MonitoringDataset(CheXpertDataset):

    def __init__(self,
                 data_path: str,
                 split: str,
                 split_config: str = 'train_valid',
                 transform: Transform = None,
                 random_transform: Transform = None,
                 max_size: int = None,
                 print_stats: bool = True,
                 plot_stats: bool = False,
                 upsample_label: str = None,
                 downsample_label: str = None,
                 # label_reader kwargs
                 label_filter: List[int] = None,
                 uncertain_to_one: Union[str, List[str]] = None,
                 uncertain_to_zero: Union[str, List[str]] = None,
                 uncertain_upper_bound: float = 0.5,
                 uncertain_lower_bound: float = 0.5,
                 one_labels: List[str] = None,
                 valid_views: List[str] = None,
                 valid_sexs: List[str] = None,
                 valid_directions: List[str] = None,
                 min_age: int = None,
                 max_age: int = None,
                 # image_reader kwargs
                 use_cache: bool = False,
                 in_memory: bool = True,
                 # monitoring dataset special kwargs
                 include_all_current_images: bool = True,
                 max_sequence_length: int = None,
                 min_sequence_length: int = None,
                 max_age_difference: int = None,
                 min_age_difference: int = None,
                 # multiprocessing kwargs
                 rank: int = 0,
                 world_size: int = 1,
                 ):
        self.include_all_current_images = include_all_current_images
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        self.max_age_difference = max_age_difference
        self.min_age_difference = min_age_difference
        
        super().__init__(
            data_path=data_path,
            split=split,
            split_config=split_config,
            transform=transform,
            random_transform=random_transform,
            max_size=max_size,
            print_stats=print_stats,
            plot_stats=plot_stats,
            upsample_label=upsample_label,
            downsample_label=downsample_label,
            # label_reader kwargs
            label_filter=label_filter,
            uncertain_to_one=uncertain_to_one,
            uncertain_to_zero=uncertain_to_zero,
            uncertain_upper_bound=uncertain_upper_bound,
            uncertain_lower_bound=uncertain_lower_bound,
            one_labels=one_labels,
            valid_views=valid_views,
            valid_sexs=valid_sexs,
            valid_directions=valid_directions,
            min_age=min_age,
            max_age=max_age,
            # image_reader kwargs
            use_cache=use_cache,
            in_memory=in_memory,
            # multiprocessing kwargs
            rank=rank,
            world_size=world_size)

    def __len__(self):
        return len(self.image_path_mapping)

    @property
    def all_metadata(self) -> List[Dict]:
        all_paths = [path for paths in self.image_path_mapping for path in paths]
        return [self.load_image_metadata(path) for path in all_paths]

    @property
    def all_labels(self) -> torch.tensor:
        labels: List[torch.tensor] = [self.label_reader[path] for path in self.label_reader.image_paths]
        labels: torch.tensor = torch.stack(labels)
        return labels

    def init_label_reader(self, **label_reader_kwargs):
        return MonitoringLabelReader(**label_reader_kwargs,
                                     include_all_current_images=self.include_all_current_images,
                                     max_sequence_length=self.max_sequence_length,
                                     min_sequence_length=self.min_sequence_length,
                                     max_age_difference=self.max_age_difference,
                                     min_age_difference=self.min_age_difference
                                     )

    def get_image_paths(self, label_reader: MonitoringLabelReader):
        # return only the keys so the splitting is done on the keys
        return list(label_reader.image_to_previous_images.keys())

    def init_image_reader(self, **image_reader_kwargs):
        _ = image_reader_kwargs.pop('image_paths')
        return PNGJPGImageReader(image_paths=self.label_reader.image_paths,
                                 **image_reader_kwargs)

    def create_image_path_mapping(self) -> Dict[ImagePath, List[ImagePath]]:
        return [previous_images + [image] for image, previous_images in self.label_reader.image_to_previous_images.items()]

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        image_paths = self.image_path_mapping[index]
        images = [self.image_reader.load_image(image_path) for image_path in image_paths]
        if self.random_transform is not None:
            images = [self.random_transform(image) for image in images]
        images = torch.stack(images)
        labels = [self.load_image_data(image_path) for image_path in image_paths]
        labels = torch.stack(labels)
        return images, labels

    def load_image_data(self, image_path) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        return self.label_reader[image_path]


def main():
    pass


if __name__ == '__main__':
    main()