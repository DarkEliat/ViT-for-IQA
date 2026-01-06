from typing import Any
import re

import numpy as np
from scipy.io import loadmat

from src.datasets.base_dataset import BaseDataset
from src.utils.data_types import Label, QualityScore
from src.utils.paths import PROJECT_ROOT


# TODO: Dodać wyciąganie nazw kolumn z pliku konfiguracyjnego .yaml zamiast ich hardcode'owania
class LiveDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config=config)

        matlab_data = loadmat(str(self.labels_path))

        self._check_label_file_keys(matlab_data=matlab_data)

        image_list = matlab_data['image_list'].squeeze()
        mos_values = matlab_data['MOS'].squeeze()

        labels = self._build_label_list(image_list=image_list, mos_values=mos_values)

        if not labels:
            raise ValueError(
                f"Error: Nie znaleziono żadnych etykiet!"
                f"Ścieżka: {self.labels_path}"
            )

        self._labels = labels


    def __len__(self):
        return len(self.labels)


    @property
    def labels(self) -> list[dict[str, Any]]:
        return self._labels


    def _get_label(self, index: int) -> Label:
        label = self.labels[index]

        return Label(
            reference_image_name=label['reference_image_name'],
            distorted_image_name=label['distorted_image_name'],
            quality_score=QualityScore(
                type='mos',
                value=label['mos'],
                normalized=False,
                model_target=False
            )
        )


    def _check_label_file_keys(self, matlab_data: dict[str, np.ndarray]) -> None:
        required_keys = {'image_list', 'MOS'}
        missing_keys = required_keys - matlab_data.keys()
        if missing_keys:
            raise KeyError(
                f"Error: W pliku z etykietami datasetu brakuje kluczy: {missing_keys}!"
                f"Ścieżka: {self.labels_path}"
            )


    def _build_label_list(self, image_list: list, mos_values: list) -> list[dict[str, Any]]:
        labels: list[dict[str, Any]] = []

        for image_name, mos_value in zip(image_list, mos_values):
            distorted_image_name = str(image_name[0])

            # Wykluczamy obrazy referencyjne z listy wszystkich obrazów
            if '_' not in distorted_image_name:
                continue

            reference_id = distorted_image_name.split("_")[0]
            candidate_reference_image_names = [
                f"{reference_id}.jpg",
                f"{reference_id}.jpeg",
                f"{reference_id}.bmp"
            ]

            reference_image_name = next(
                (
                    candidate
                    for candidate in candidate_reference_image_names
                    if self.reference_images_map.has_file_path(candidate)
                ),
                None
            )

            if reference_image_name is None:
                raise FileNotFoundError(
                    f"Error: Nie znaleziono obrazu referencyjnego dla: {distorted_image_name}!\n"
                    f"Ścieżka: {self.reference_images_path}"
                )

            labels.append({
                'reference_image_name': reference_image_name,
                'distorted_image_name': distorted_image_name,
                'mos': float(mos_value),
            })

        return labels