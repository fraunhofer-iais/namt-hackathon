from typing import List, Tuple, Dict, Optional, Union, Any
import os
import glob
import csv
import random


from collections import defaultdict

import numpy as np
from key2med.utils.logging import tqdm


import torch
from key2med.data.patients import Patient, Study

import logging
import coloredlogs

coloredlogs.install(level=logging.INFO)
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Path = str
ImagePath = str
Label = int


class LabelReader:
    data: Dict[ImagePath, Any] = None
    image_paths: List[ImagePath] = None


class ChexpertLabelReader(LabelReader):
    sex_values = {
        'Male': 0,
        'Female': 1,
        'Unknown': 2
    }
    view_values = {
        'Frontal': 0,
        'Lateral': 1
    }
    direction_values = {
        'AP': 0,
        'PA': 1,
        'LL': 2,
        'RL': 3,
        '': 4
    }

    def __init__(self,
                 label_file,
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
                 max_size: int = None):
        self.label_file = label_file
        self.max_size = max_size

        self.valid_views: List[str] = valid_views or list(self.view_values.keys())
        self.valid_directions: List[str] = valid_directions or list(self.direction_values.keys())
        self.valid_sexs: List[str] = valid_sexs or list(self.sex_values.keys())
        self.min_age = min_age or 0
        self.max_age = max_age or 100

        self.uncertain_to_one, self.uncertain_to_zero = self.init_uncertain_mapping(uncertain_to_one, uncertain_to_zero)
        self.uncertain_upper_bound, self.uncertain_lower_bound = uncertain_upper_bound, uncertain_lower_bound

        self.data, self.index_to_label = self.read_label_csv(label_file, label_filter)
        self.label_to_index = {label: index for index, label in enumerate(self.index_to_label)}

        self.one_indices: List[int] = self.init_one_indices(one_labels)

        self.data = self.filter_data(self.data)
        self.image_paths = list(self.data.keys())

    def init_one_indices(self, one_labels: List[str]) -> List[int]:
        if not one_labels:
            return []
        return [self.label_to_index[label] for label in one_labels]

    def __getitem__(self, path):
        data = self.data[path]
        return self.prepare_data(data)

    @property
    def label_dim(self) -> Optional[int]:
        """
        Number of labels in the dataset.
        Default CheXpert 14, competition mode 5.
        :return: int Number of labels
        """
        return len(self.index_to_label)

    def filter_data(self, data: Dict[ImagePath, Dict]):
        filtered_data = {}

        for image_path, image_data in data.items():
            if image_data['view'] not in self.valid_views:
                continue
            if image_data['direction'] not in self.valid_directions:
                continue
            if image_data['sex'] not in self.valid_sexs:
                continue
            if not self.min_age <= image_data['age'] <= self.max_age:
                continue
            if self.one_indices and not any([data['labels'][label_index] == 1. for label_index in self.one_indices]):
                continue
            filtered_data[image_path] = image_data
        return filtered_data

    def prepare_data(self, data: Dict) -> torch.Tensor:
        # sex = self.sex_values[data['sex']]
        # view = self.view_values[data['view']]
        # direction = self.direction_values[data['direction']
        labels = self.convert_labels_live(data['labels'])
        return labels

    def convert_labels_live(self, labels: torch.tensor) -> torch.tensor:
        labels = torch.tensor(
            [self.convert_uncertain_labels(label, self.index_to_label[i]) for i, label in enumerate(labels)])
        return labels

    def convert_uncertain_labels(self, label: torch.tensor, label_name: str):
        if label != -1.0:
            return label

        if label_name in self.uncertain_to_zero:
            return 0.0
        if label_name in self.uncertain_to_one:
            return 1.0
        if self.uncertain_upper_bound != self.uncertain_lower_bound:
            return random.uniform(self.uncertain_lower_bound, self.uncertain_upper_bound)
        return self.uncertain_lower_bound

    @staticmethod
    def init_uncertain_mapping(uncertain_to_one: Optional[Union[str, List[str]]],
                               uncertain_to_zero: Optional[Union[str, List[str]]]):
        if uncertain_to_one is None:
            uncertain_to_one = []
        elif uncertain_to_one == 'best':
            uncertain_to_one = ['Edema', 'Atelectasis']

        if uncertain_to_zero is None:
            uncertain_to_zero = []
        elif uncertain_to_zero == 'best':
            uncertain_to_zero = ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']

        return uncertain_to_one, uncertain_to_zero

    def init_label_filter(self, label_filter: Union[List[int], str], label_names: List[str]) -> Optional[List[int]]:
        if label_filter is None or label_filter == 'full':
            return None
        if label_filter == 'competition':
            return [2, 5, 6, 8, 10]
        if isinstance(label_filter, list) and isinstance(label_filter[0], int):
            return label_filter
        if isinstance(label_filter, list) and isinstance(label_filter[0], str):
            return [index for index, label_name in enumerate(label_names) if label_name in label_filter]
        raise NotImplementedError

    def read_label_csv(self, file, label_filter) -> Tuple[Dict, List[str]]:
        """
        Read CheXpert label csv.
        :param file: Path to .csv
        :param label_filter: xx
        :return: Tuple[Dict, List[str]] Label data dictionary and list of label names.
        """
        data: Dict[ImagePath, Dict] = {}
        with open(file, 'r') as f:
            reader = csv.reader(f)
            label_names = next(reader)[5:]
            label_filter: List[int] = self.init_label_filter(label_filter, label_names)
            logger.info(f'Found labels in {file}: {label_names}')
            for index, row in tqdm(enumerate(reader), desc=f'Reading label csv file {file}'):
                image_path = row[0]
                data[image_path] = self.read_row(row, label_names, label_filter)
                if self.max_size is not None and index > self.max_size:
                    break
        if label_filter is not None:
            label_names = [label_names[i] for i in label_filter]
        return data, label_names

    def read_row(self, row: List, label_names: List[str], label_filter: List[int]) -> Dict[str, Any]:
        """
        Read a single row in the label csv. Convert the labels.
        :param row: List[str]
        :param label_filter: xx
        :return: Dict
        """
        return {
            'sex': row[1],
            'age': int(row[2]),
            'view': row[3],
            'direction': row[4],
            'labels': self.convert_labels(row[5:], label_names, label_filter)
        }

    def convert_labels(self, labels: List[str], label_names: List[str], label_filter: List[int]) -> torch.Tensor:
        """
        Label conversion while reading the label file.
        Only done once before training. No random transformations here.

        Possible labels:
        '1.0': positive
        '0.0': negative
        '-1.0': uncertain
        '': no mention

        :param labels: Labels from row of .csv file. As strings.
        :param label_filter: xx
        :return: torch.Tensor or list of floats. Initially converted labels.
        """
        convert = {
            '1.0': 1.0,
            '0.0': 0.0,
            '': 0.0,
            '-1.0': -1.0
        }
        labels = torch.FloatTensor([convert[x] for x in labels])
        if label_filter is not None:
            labels = labels[label_filter]
        return labels


class PatientLabelReader(ChexpertLabelReader):
    """
    Sorts images by patient and by study.
    Outputs label and view and direction for each image
    """

    def __init__(self, **label_reader_kwargs):
        super().__init__(**label_reader_kwargs)

        self.split_name = 'train' if 'train' in self.image_paths[0] else 'valid'
        self.patients = self.patients_from_paths(self.image_paths)
        self.studies = [study for patient in self.patients for study in patient.studies]

    def patients_from_paths(self, image_paths) -> List[Patient]:
        """
        From a list of image paths, create Patient object.
        Sort each image path by study and patient and merge into Patient objects
        """
        patient_studies_to_paths = defaultdict(list)
        for path in image_paths:
            patient_study = '/'.join(self.split_path(path, self.split_name))
            patient_studies_to_paths[patient_study].append(path)
        patients_to_studies = defaultdict(list)
        for patient_study, paths in patient_studies_to_paths.items():
            patient, study = patient_study.split('/')
            patients_to_studies[patient].append(
                Study(name=study, images=sorted(paths, key=lambda x: self.study_number_from_path(x))))
        patients = []
        for patient, studies in patients_to_studies.items():
            patients.append(Patient(name=patient, studies=sorted(studies, key=lambda x: x.number)))
        return patients

    @staticmethod
    def split_path(path: str, split: str = 'train') -> Tuple[str, str]:
        _, path = path.split(split)
        _, patient, study, filename = path.split('/')
        return patient, study

    def study_number_from_path(self, path):
        return int(path.split('study')[-1].split('/')[0])

    def prepare_data(self, data: Dict) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        # sex = self.sex_values[data['sex']]
        view = self.view_values[data['view']]
        direction = self.direction_values[data['direction']]
        labels = self.convert_labels_live(data['labels'])
        return labels, torch.tensor(view), torch.tensor(direction)


class MonitoringLabelReader(PatientLabelReader):
    """
    Additionally to PatientLabelReader, sorts images by study in chronological order
    and implements a max_age and min_age difference between study images that belong together.
    """

    def __init__(self,
                 include_all_current_images: bool = True,
                 max_sequence_length: int = None,
                 min_sequence_length: int = None,
                 max_age_difference: int = None,
                 min_age_difference: int = None,
                 **label_reader_kwargs):
        super().__init__(**label_reader_kwargs)
        self.include_all_current_images = include_all_current_images
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        self.max_age_difference = max_age_difference or 100
        self.min_age_difference = min_age_difference or 0

        self.image_to_previous_images = self.create_previous_images_mapping(self.patients)
        # we have filtered the images when creating the mapping
        # now load only the relevant images
        self.image_paths = [image for images in self.image_to_previous_images.values() for image in images]
        self.image_paths += list(self.image_to_previous_images.keys())
        self.image_paths = list(set(self.image_paths))

    def age_from_study(self, study: Study) -> int:
        return self.data[study.images[0]]['age']

    def filter_studies_by_age(self, studies, current_age):
        output = []
        for study in studies:
            age_diff = current_age - self.age_from_study(study)
            if self.min_age_difference <= age_diff <= self.max_age_difference:
                output.append(study)
        return output

    def create_previous_images_mapping(self,
                                       patients: List[Patient]):
        image_to_previous_images: Dict[ImagePath, List[ImagePath]] = {}
        for patient in patients:
            for index, current_study in enumerate(patient.studies):
                previous_studies = patient.studies[:index]
                current_age = self.age_from_study(current_study)
                previous_studies = self.filter_studies_by_age(previous_studies, current_age)
                for image_index, image in enumerate(current_study.images):
                    previous_images = [image for study in previous_studies for image in study.images]
                    if self.include_all_current_images:
                        previous_images += current_study.images[:image_index]
                        previous_images += current_study.images[image_index + 1:]
                    if self.min_sequence_length is not None:
                        if len(previous_images) <= self.min_sequence_length:
                            continue
                    if self.max_sequence_length is not None:
                        previous_images = previous_images[-self.max_sequence_length:]
                    image_to_previous_images[image] = previous_images
        return image_to_previous_images

    def prepare_data(self, data: Dict) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        # sex = self.sex_values[data['sex']]
        # view = self.view_values[data['view']]
        # direction = self.direction_values[data['direction']]
        labels = self.convert_labels_live(data['labels'])
        return labels#, torch.tensor(view), torch.tensor(direction)
