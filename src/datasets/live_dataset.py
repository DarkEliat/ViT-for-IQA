from typing import Any

import numpy as np
from scipy.io import loadmat

from src.datasets.base_dataset import BaseDataset
from src.utils.data_types import Label, QualityScore, DatasetConfig, ModelConfig


class LiveDataset(BaseDataset[ list[dict[str, Any]] ]):
    def __init__(self, dataset_config: DatasetConfig, model_config: ModelConfig):
        super().__init__(dataset_config=dataset_config, model_config=model_config)

        self._labels_container = self._build_labels_container()


    @property
    def labels_container(self) -> list[dict[str, Any]]:
        return self._labels_container


    def _check_label_file_keys(self, matlab_data: dict[str, np.ndarray]) -> None:
        required_keys = {'image_list', 'MOS'}
        missing_keys = required_keys - matlab_data.keys()
        if missing_keys:
            raise KeyError(
                f"Error: W pliku z etykietami datasetu brakuje kluczy: {missing_keys}!"
                f"Ścieżka: {self.labels_path}"
            )


    def get_reference_image_name(self, distorted_image_name: str) -> str:
        reference_prefix = distorted_image_name.split(sep='_', maxsplit=1)[0]
        candidate_reference_image_names = [
            f"{reference_prefix}.jpg",
            f"{reference_prefix}.jpeg",
            f"{reference_prefix}.bmp"
        ]

        reference_image_name = next(
            (
                candidate
                for candidate in candidate_reference_image_names
                if self.reference_images_map.has_file_name(candidate)
            ),
            None
        )

        if reference_image_name is None:
            raise FileNotFoundError(
                f"Error: Nie znaleziono obrazu referencyjnego dla: {distorted_image_name}!\n"
                f"Ścieżka: {self.reference_images_path}"
            )

        return reference_image_name


    def get_reference_image_names_per_distorted_image(self) -> list[str]:
        return [
            str(label_dict['reference_image_name'])
            for label_dict in self.labels_container
        ]


    def _build_labels_container(self) -> list[dict[str, Any]]:
        matlab_data = loadmat(str(self.labels_path))

        self._check_label_file_keys(matlab_data=matlab_data)

        # TODO: Dodać wyciąganie nazw kolumn z pliku konfiguracyjnego .yaml zamiast ich hardcode'owania
        image_list = matlab_data['image_list'].squeeze()
        mos_values = matlab_data['MOS'].squeeze()

        labels_container: list[dict[str, Any]] = []

        for image_name, mos_value in zip(image_list, mos_values):
            distorted_image_name = str(image_name[0])

            # Wykluczamy obrazy referencyjne z listy wszystkich obrazów
            if '_' not in distorted_image_name:
                continue

            reference_image_name = self.get_reference_image_name(distorted_image_name=distorted_image_name)

            labels_container.append({
                'reference_image_name': reference_image_name,
                'distorted_image_name': distorted_image_name,
                'quality_score': float(mos_value),
            })

        if not labels_container:
            raise ValueError(
                f"Error: Nie znaleziono żadnych etykiet!"
                f"Ścieżka: {self.labels_path}"
            )

        return labels_container


    def get_label(self, index: int) -> Label:
        label = self.labels_container[index]

        return Label(
            reference_image_name=label['reference_image_name'],
            distorted_image_name=label['distorted_image_name'],
            quality_score=QualityScore(
                type='mos',
                min_value=self.dataset_config['dataset']['quality_label']['min'],
                max_value=self.dataset_config['dataset']['quality_label']['max'],
                value=label['quality_score'],
                normalized=False,
                model_target=False
            )
        )
