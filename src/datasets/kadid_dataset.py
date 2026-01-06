import pandas as pd

from src.datasets.base_dataset import BaseDataset
from src.utils.data_types import Label, QualityScore


class Kadid10kDataset(BaseDataset[ pd.DataFrame ]):
    def __init__(self, config):
        super().__init__(config=config)

        self._labels = self._build_labels()


    @property
    def labels(self) -> pd.DataFrame:
        return self._labels


    def _extract_reference_image_name(self, distorted_image_name: str) -> str:
        matches = self.labels.loc[
            self.labels['distorted_image_name'] == distorted_image_name,
            'reference_image_name'
        ]

        if matches.empty:
            raise KeyError(f"Error: Nie znaleziono obrazu referencyjnego dla obrazu `{distorted_image_name}`!")

        if len(matches) > 1:
            raise ValueError(f"Error: Znaleziono wiele (zamiast TYLKO jednego) obrazów referencyjnych dla obrazu `{distorted_image_name}`!")

        return matches.iloc[0]


    def _build_labels(self) -> pd.DataFrame:
        labels = pd.read_csv(
            self.labels_path,
            sep=r',',
            header=0
        )
        labels.columns = labels.columns.str.strip()

        labels.astype({
            'dist_img': str,
            'ref_img': str,
            'dmos': float,
            'var': float
        })

        labels.columns = (
            labels.columns
            .str.strip()
            .str.replace("\\\\", "/", regex=False)
            .str.replace("\\", "/", regex=False)
        )

        labels = labels.rename(
            columns={
                'ref_img': 'reference_image_name',
                'dist_img': 'distorted_image_name',
                'dmos': 'quality_score'
            }
        )

        if labels.isnull().any().any():
            raise ValueError(
                f"Error: Plik z etykietami zawiera puste wartości po czyszczeniu!"
                f"Ścieżka: {self.labels_path}"
            )

        return labels


    def _get_label(self, index: int) -> Label:
        label = self.labels.iloc[index]

        return Label(
            reference_image_name=label['reference_image_name'],
            distorted_image_name=label['distorted_image_name'],
            quality_score=QualityScore(
                type='dmos',
                value=label['quality_score'],
                normalized=False,
                model_target=False
            )
        )
