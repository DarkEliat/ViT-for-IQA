from pathlib import Path

import pandas as pd
from PIL import Image
import torch

from torch.utils.data import Dataset as TorchDataset
from src.utils.image_preprocessing import resize
from src.utils.paths import PROJECT_ROOT


class BaseDataset(TorchDataset):
    def __init__(self, config):
        self.config = config

        labels_path = PROJECT_ROOT / config['dataset']['labels_path']
        self.labels = pd.read_csv(labels_path)
        self.reference_images_path = Path(config['dataset']['reference_images_path'])
        self.distorted_images_path = Path(config['dataset']['distorted_images_path'])

        self.target_image_size = (
            config['model']['input']['image_size']['width'],
            config['model']['input']['image_size']['height'],
        )

        self.keep_original_aspect_ratio = config['model']['input']['keep_original_aspect_ratio']


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, index):
        row = self.labels.iloc[index]

        # TODO: Dodać wyciąganie nazw kolumn z pliku konfiguracyjnego .yaml zamiast ich hardcode'owania
        reference_image_name = row['ref_img']
        distorted_image_name = row['dist_img']
        dmos_value = row['dmos']

        reference_image_path = self.reference_images_path / reference_image_name
        distorted_image_path = self.distorted_images_path / distorted_image_name

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

        return reference_image, distorted_image, torch.tensor(dmos_value, dtype=torch.float32)
