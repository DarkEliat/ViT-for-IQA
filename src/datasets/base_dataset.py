from abc import abstractmethod, ABC
from pathlib import Path
from typing import Any, Generic, TypeVar

import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms

from src.datasets.file_map import FileMap
from src.utils.data_types import Label, Config, UnifiedQualityScore
from src.utils.image_preprocessing import resize
from src.utils.paths import PROJECT_ROOT


LabelsContainerType = TypeVar('LabelsContainerType')


class BaseDataset(ABC, TorchDataset, Generic[LabelsContainerType]):
    def __init__(self, config):
        self._config = config

        self._labels_path = PROJECT_ROOT / config['dataset']['labels_path']

        self._reference_images_path = Path(config['dataset']['reference_images_path'])
        self._distorted_images_path = Path(config['dataset']['distorted_images_path'])

        self._reference_images_map = FileMap(files_directory_path=self.reference_images_path)
        self._distorted_images_map = FileMap(files_directory_path=self.distorted_images_path)

        self._target_image_size = (
            config['model']['input']['image_size']['width'],
            config['model']['input']['image_size']['height'],
        )

        self._keep_original_aspect_ratio = config['model']['input']['keep_original_aspect_ratio']

        self._transform = transforms.Compose([
            transforms.ToTensor()
        ])


    def __getitem__(self, index):
        label = self._get_label(index)

        reference_image_name = label['reference_image_name']
        distorted_image_name = label['distorted_image_name']
        quality_score_value = label['quality_score']['value']

        quality_score_value = self._unify_quality_score(value=quality_score_value).value

        reference_image_path = self.reference_images_map.get_file_path(reference_image_name)
        distorted_image_path = self.distorted_images_map.get_file_path(distorted_image_name)

        reference_image = Image.open(reference_image_path).convert('RGB')
        distorted_image = Image.open(distorted_image_path).convert('RGB')

        reference_image = resize(
            image=reference_image,
            target_size=self.target_image_size,
            keep_aspect_ratio=self.keep_original_aspect_ratio
        )
        distorted_image = resize(
            image=distorted_image,
            target_size=self.target_image_size,
            keep_aspect_ratio=self.keep_original_aspect_ratio
        )

        reference_image = self.transform(reference_image)
        distorted_image = self.transform(distorted_image)

        return reference_image, distorted_image, torch.tensor(quality_score_value, dtype=torch.float32)


    def __len__(self):
        return len(self.labels)


    @property
    @abstractmethod
    def labels(self) -> LabelsContainerType: ...

    @property
    def labels_path(self) -> Path:
        return self._labels_path

    @property
    def config(self) -> Config:
        return self._config

    @property
    def reference_images_path(self) -> Path:
        return self._reference_images_path

    @property
    def distorted_images_path(self) -> Path:
        return self._distorted_images_path

    @property
    def reference_images_map(self) -> FileMap:
        return self._reference_images_map

    @property
    def distorted_images_map(self) -> FileMap:
        return self._distorted_images_map

    @property
    def target_image_size(self) -> tuple[int, int]:
        return self._target_image_size

    @property
    def keep_original_aspect_ratio(self) -> bool:
        return self._keep_original_aspect_ratio

    @property
    def transform(self) -> transforms.Compose:
        return self._transform


    @abstractmethod
    def _unify_quality_score(self, value: float) -> UnifiedQualityScore: ...

    @abstractmethod
    def _extract_reference_image_name(self, distorted_image_name: str) -> str: ...

    @abstractmethod
    def _build_labels(self) -> LabelsContainerType: ...

    @abstractmethod
    def _get_label(self, index: int) -> Label: ...
