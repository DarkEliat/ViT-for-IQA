import pandas as pd
from PIL import Image
import torch

from src.datasets.base_dataset import BaseDataset
from src.utils.data_types import Label, QualityScore
from src.utils.image_preprocessing import resize
from src.utils.paths import PROJECT_ROOT


class Kadid10kDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config=config)

        self._labels = pd.read_csv(self.labels_path)
        self._labels.columns = self._labels.columns.str.strip()

        for column_name in self._labels.select_dtypes(include='object').columns:
            self._labels[column_name] = (self._labels[column_name].str.strip()
                                         .str.replace("\\\\", "/", regex=False)
                                         .str.replace("\\", "/", regex=False))

        if self._labels.isnull().any().any():
            raise ValueError("Error: Plik z etykietami zawiera puste wartości po czyszczeniu!")


    def __len__(self):
        return len(self.labels)


    @property
    def labels(self) -> pd.DataFrame:
        return self._labels


    def _get_label(self, index: int) -> Label:
        label = self.labels.iloc[index]

        # TODO: Dodać wyciąganie nazw kolumn z pliku konfiguracyjnego .yaml zamiast ich hardcode'owania
        return Label(
            reference_image_name=label['ref_img'],
            distorted_image_name=label['dist_img'],
            quality_score=QualityScore(
                type='dmos',
                value=label['dmos'],
                normalized=False,
                model_target=False
            )
        )
