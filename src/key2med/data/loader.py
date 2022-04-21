from typing import List, Union, Optional, Callable, Any, Tuple
import logging
import coloredlogs

coloredlogs.install(level=logging.INFO)
logging.basicConfig()
from abc import ABC
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader

from key2med.utils.transforms import Transform, ColorRandomAffineTransform, ColorTransform, ResizeTransform
from key2med.data.datasets import BaseDataset, CheXpertDataset, StudyDataset, MonitoringDataset


class ADataLoader(ABC):
    """
    Abstract base class for all dataloaders.
    Defines the train, validate and test dataloaders with default None.
    """

    @property
    def train(self):
        return None

    @property
    def validate(self):
        return None

    @property
    def test(self):
        return None

    @property
    def n_train_batches(self):
        if self.train is None:
            return 0
        return len(self.train)

    @property
    def n_validate_batches(self):
        if self.validate is None:
            return 0
        return len(self.validate)

    @property
    def n_test_batches(self):
        if self.test is None:
            return 0
        return len(self.test)


class BaseDataLoader(ADataLoader):
    """
    Basic dataloader class.
    To be called with a dataset to be split or already split datasets.
    Creates torch dataloaders for the provided datasets.
    """

    def __init__(self,
                 dataset: BaseDataset = None,
                 valid_size: float = 0.1,
                 test_size: float = 0.1,
                 train_dataset: BaseDataset = None,
                 valid_dataset: BaseDataset = None,
                 test_dataset: BaseDataset = None,
                 batch_size: int = None,
                 collate_function: Optional[Callable] = None,
                 n_workers: int = 1):
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.collate_function = collate_function

        self.train_set_size = 0
        self.__train_loader = None
        self.valid_set_size = 0
        self.__valid_loader = None
        self.test_set_size = 0
        self.__test_loader = None

        if all([x is None for x in [train_dataset, valid_dataset, test_dataset]]):
            self.init_split_from_dataset(dataset, valid_size, test_size)
        else:
            self.init_split_from_split(train_dataset, valid_dataset, test_dataset)

    @staticmethod
    def all_are_none(*args):
        return all([x is None for x in args])

    def init_split_from_dataset(self, dataset, valid_size, test_size):
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split_valid = int(np.floor(valid_size * dataset_size))
        split_test = split_valid + int(np.floor(test_size * dataset_size))
        np.random.shuffle(indices)

        train_indices = indices[split_test:]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        self.train_set_size = len(train_sampler)
        self.__train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                          shuffle=False, num_workers=self.n_workers,
                                                          collate_fn=self.collate_function,
                                                          sampler=train_sampler)
        print(f'{self.train_set_size:,} samples for training')

        valid_indices = indices[:split_valid]
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)
        self.valid_set_size = len(valid_sampler)
        self.__valid_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                          shuffle=False, num_workers=self.n_workers,
                                                          collate_fn=self.collate_function,
                                                          sampler=valid_sampler)
        print(f'{self.valid_set_size:,} samples for validation')

        test_indices = indices[split_valid:split_test]
        test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)
        self.test_set_size = len(test_sampler)
        self.__test_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                                         shuffle=False, num_workers=self.n_workers,
                                                         collate_fn=self.collate_function,
                                                         sampler=test_sampler)
        print(f'{self.test_set_size:,} samples for validation')

    def init_split_from_split(self, train_dataset, valid_dataset, test_dataset):
        if train_dataset is not None:
            self.train_set_size = len(train_dataset)
            self.__train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                                              collate_fn=self.collate_function,
                                                              shuffle=True, num_workers=self.n_workers)
            print(f'{self.train_set_size:,} samples for training')
        else:
            self.train_set_size = 0
            self.__train_loader = None

        if valid_dataset is not None:

            self.valid_set_size = len(valid_dataset)
            self.__valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size,
                                                              collate_fn=self.collate_function,
                                                              shuffle=True, num_workers=self.n_workers)
            print(f'{self.valid_set_size:,} samples for validation')
        else:
            self.valid_set_size = 0
            self.__valid_loader = None

        if test_dataset is not None:
            self.test_set_size = len(test_dataset)
            self.__test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size,
                                                             collate_fn=self.collate_function,
                                                             shuffle=True, num_workers=self.n_workers)
            print(f'{self.test_set_size:,} samples for testing')
        else:
            self.test_set_size = 0
            self.__test_loader = None

    @property
    def train(self):
        return self.__train_loader

    @property
    def validate(self):
        return self.__valid_loader

    @property
    def test(self):
        return self.__test_loader


class CheXpertDataLoader(BaseDataLoader):
    """
    Dataloader for CheXpert dataset.
    When called with a valid path to the chexpert dataset, initializes
    the training and validation split given by chexpert.
    Most arguments given to init are passed on to chexpert datasets.
    """

    def __init__(self,
                 data_path,
                 batch_size: int = 16,
                 img_resize: int = 224,
                 channels: int = 3,
                 # dataset kwargs
                 split_config: str = 'train_valid',
                 transform: Transform = None,
                 do_random_transform: bool = True,
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
                 # dataloader kwargs
                 rank: int = 0,
                 world_size: int = 1,
                 collate_function: Optional[Callable] = None,
                 n_workers: int = 0):
        """

        :param data_path: Path to chexpert dataset.
        :param batch_size: Batch size for both training and validation.
        :param img_resize: Dimension of image as input to the model.
        :param channels: Number of image channels. CheXpert are B/W-images by default, one channel. Most pretrained vision models
                         expect 3 channels for color. By default, this class implements a transform that copies the grayscale values
                         from one channel to 3 channels.
        :param transform: Transform class object to transform the data before training. By default resizes the image to img_resize.
        :param uncertainty_upper_bound: Uncertain values (if not in uncertain_to_one or uncertain_to_zero) are mapped onto an interval
                                        between uncertainty_upper_bound and uncertainty_lower_bound.
                                        To map onto exactly one value set both values the same.
        :param uncertainty_lower_bound: Uncertain values (if not in uncertain_to_one or uncertain_to_zero) are mapped onto an interval
                                        between uncertainty_upper_bound and uncertainty_lower_bound.
                                        To map onto exactly one value set both values the same.
        :param one_labels: Labels that have to be one, or the datapoint is filtered. List of integers, refers to the list of labels AFTER
                           filtering by label_filter.
        :param uncertain_to_one: Labels for which uncertain values should be mapped onto 1.0. List of string with the label name.
        :param uncertain_to_zero: Labels for which uncertain values should be mapped onto 0.0. List of string with the label name.
        :param do_random_transform: Boolean, do a random transform when loading a training image.
        :param min_age: Minimum age of person in xray.
        :param max_age: Maximum age of person in xray.
        :param sex_values: One or multiple of ['Unknown', 'Male', 'Female']
        :param frontal_lateral_values: One or multiple of ['Frontal', 'Lateral']
        :param ap_pa_values: One or multiple of ['', 'AP', 'PA', 'LL', 'RL']
        :param label_filter: Set to 'competition' to only train and evaluate on the competition classes:
                             'Edema', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Pleural Effusion'
        :param n_worker: Number of processes for the torch dataloader
        :param max_size: Limit for number of images in training and validation each. For debugging.
        :param in_memory: Load the entire dataset into memory before starting the training.
        :param use_cache: Find or create a cache file for the datasets and load from there.
        """

        self.image_dim = img_resize
        self.channels = channels
        self.uncertain_upper_bound = uncertain_upper_bound
        self.uncertain_lower_bound = uncertain_lower_bound
        self.do_random_transform = do_random_transform

        base_kwargs = {
            "data_path": data_path,
            "split_config": split_config,
            "transform": transform or self.transform,
            "max_size": max_size,
            "print_stats": print_stats,
            "plot_stats": plot_stats,
            "label_filter": label_filter,
            "uncertain_to_one": uncertain_to_one,
            "uncertain_to_zero": uncertain_to_zero,
            "uncertain_upper_bound": uncertain_upper_bound,
            "uncertain_lower_bound": uncertain_lower_bound,
            "one_labels": one_labels,
            "valid_views": valid_views,
            "valid_sexs": valid_sexs,
            "valid_directions": valid_directions,
            "min_age": min_age,
            "max_age": max_age,
            "use_cache": use_cache,
            "in_memory": in_memory,
            "channels": self.channels,
        }
        
        train_dataset = self.init_dataset(**base_kwargs,
                                          split='train',
                                          random_transform=self.random_transform,
                                          upsample_label=upsample_label,
                                          downsample_label=downsample_label,
                                          rank=rank,
                                          world_size=world_size)

        valid_dataset = self.init_dataset(**base_kwargs,
                                          split='valid',
                                          random_transform=None,
                                          upsample_label=None,
                                          downsample_label=None,
                                          rank=0,
                                          world_size=1
                                          )

        test_dataset = self.init_dataset(**base_kwargs,
                                         split='test',
                                         random_transform=None,
                                         upsample_label=None,
                                         downsample_label=None,
                                         rank=0,
                                         world_size=1
                                         )

        self.index_to_label = train_dataset.index_to_label
        self.collate_function = collate_function or self.default_collate_function

        super().__init__(train_dataset=train_dataset,
                         valid_dataset=valid_dataset,
                         test_dataset=test_dataset,
                         batch_size=batch_size,
                         collate_function=self.collate_function,
                         n_workers=n_workers)

    def init_dataset(self, *args, **kwargs):
        if kwargs.get('split_config') != 'train_valid_test' and kwargs.get('split') == 'test':
            return None
        return CheXpertDataset(*args, **kwargs)

    @property
    def transform(self):
        return ColorTransform(self.image_dim)

    @property
    def random_transform(self):
        if self.do_random_transform:
            return ColorRandomAffineTransform()
        else:
            return None

    @property
    def default_collate_function(self):
        # torch default collate function will be used
        return None

    @property
    def label_dim(self):
        if self.train is not None:
            return self.train.dataset.label_dim
        if self.validate is not None:
            return self.validate.dataset.label_dim
        if self.test is not None:
            return self.test.dataset.label_dim

    @property
    def imratio(self):
        return self.train.dataset.imratio

    @property
    def imratios(self):
        return self.train.dataset.imratios


# CheXpertDataloader now color by default
ColorCheXpertDataLoader = CheXpertDataLoader


class StudyDataLoader(ColorCheXpertDataLoader):

    def init_dataset(self, *args, **kwargs):
        if kwargs.get('split_config') != 'train_valid_test' and kwargs.get('split') == 'test':
            return None
        return StudyDataset(*args, **kwargs)

    @property
    def default_collate_function(self):
        return collate_studies_function


class MonitoringDataLoader(ColorCheXpertDataLoader):

    def __init__(self, *args,
                 include_all_current_images: bool = True,
                 max_sequence_length: int = None,
                 min_sequence_length: int = None,
                 max_age_difference: int = None,
                 min_age_difference: int = None,
                 **kwargs):
        self.include_all_current_images = include_all_current_images
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        self.max_age_difference = max_age_difference
        self.min_age_difference = min_age_difference
        super().__init__(*args, **kwargs)

    def init_dataset(self, *args, **kwargs):
        if kwargs.get('split_config') != 'train_valid_test' and kwargs.get('split') == 'test':
            return None
        if kwargs.get('split_config') == 'train_valid' and kwargs.get('split') == 'valid':
            return None
        return MonitoringDataset(*args,
                                 max_sequence_length=self.max_sequence_length,
                                 min_sequence_length=self.min_sequence_length,
                                 max_age_difference=self.max_age_difference,
                                 min_age_difference=self.min_age_difference,
                                 **kwargs)

    @property
    def default_collate_function(self):
        return collate_monitoring_function


def collate_monitoring_function(batch) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]:
    images = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    lens = [img.shape[0] for img in images]
    return images, labels, lens


def collate_studies_function(batch) -> Tuple[
    List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    images = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    views = [x[2] for x in batch]
    directions = [x[3] for x in batch]
    return images, labels, views, directions


def main():
    pass


if __name__ == '__main__':
    main()
