from typing import Any

import numpy as np
from scipy.io import loadmat

from src.datasets.base_dataset import BaseDataset
from src.utils.data_types import Label, QualityScore


class LiveDataset(BaseDataset[ list[dict[str, Any]] ]):
    def __init__(self, config):
        super().__init__(config=config)

        self._labels = self._build_labels()


    @property
    def labels(self) -> list[dict[str, Any]]:
        return self._labels


    def _check_label_file_keys(self, matlab_data: dict[str, np.ndarray]) -> None:
        required_keys = {'image_list', 'MOS'}
        missing_keys = required_keys - matlab_data.keys()
        if missing_keys:
            raise KeyError(
                f"Error: W pliku z etykietami datasetu brakuje kluczy: {missing_keys}!"
                f"Ścieżka: {self.labels_path}"
            )


    def _extract_reference_image_name(self, distorted_image_name: str) -> str:
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


    def _build_labels(self) -> list[dict[str, Any]]:
        matlab_data = loadmat(str(self.labels_path))

        self._check_label_file_keys(matlab_data=matlab_data)

        # TODO: Dodać wyciąganie nazw kolumn z pliku konfiguracyjnego .yaml zamiast ich hardcode'owania
        image_list = matlab_data['image_list'].squeeze()
        mos_values = matlab_data['MOS'].squeeze()

        labels: list[dict[str, Any]] = []

        for image_name, mos_value in zip(image_list, mos_values):
            distorted_image_name = str(image_name[0])

            # Wykluczamy obrazy referencyjne z listy wszystkich obrazów
            if '_' not in distorted_image_name:
                continue

            reference_image_name = self._extract_reference_image_name(distorted_image_name=distorted_image_name)

            labels.append({
                'reference_image_name': reference_image_name,
                'distorted_image_name': distorted_image_name,
                'quality_score': float(mos_value),
            })

            if not labels:
                raise ValueError(
                    f"Error: Nie znaleziono żadnych etykiet!"
                    f"Ścieżka: {self.labels_path}"
                )

        return labels


    def _get_label(self, index: int) -> Label:
        label = self.labels[index]

        return Label(
            reference_image_name=label['reference_image_name'],
            distorted_image_name=label['distorted_image_name'],
            quality_score=QualityScore(
                type='mos',
                value=label['quality_score'],
                normalized=False,
                model_target=False
            )
        )
