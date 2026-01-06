import pandas as pd

from src.datasets.base_dataset import BaseDataset
from src.utils.data_types import Label, QualityScore


class TidDataset(BaseDataset[ pd.DataFrame ]):
    def __init__(self, config):
        super().__init__(config=config)

        self._labels = self._build_labels()


    @property
    def labels(self) -> pd.DataFrame:
        return self._labels


    def _extract_reference_image_name(self, distorted_image_name: str) -> str:
        reference_prefix = distorted_image_name.split(sep='_', maxsplit=1)[0]

        reference_image_name = f"{reference_prefix}.bmp"

        if not self.reference_images_map.has_file_name(file_name=reference_image_name):
            raise FileNotFoundError(
                f"Error: Nie znaleziono obrazu referencyjnego dla: {distorted_image_name}!\n"
                f"Ścieżka: {self.reference_images_path}"
            )

        return reference_image_name


    def _build_labels(self) -> pd.DataFrame:
        labels: pd.DataFrame = pd.read_csv(
            self.labels_path,
            sep=r'\s+',  # Dowolna liczba białych znaków jako separator kolumn
            header=None,
            names=['quality_score', 'distorted_image_name'],
            dtype={
                'quality_score': float,
                'distorted_image_name': str
            }
        )
        labels.columns = labels.columns.str.strip()

        for column_name in labels.select_dtypes(include='object').columns:
            labels[column_name] = (labels[column_name].str.strip()
                                         .str.replace("\\\\", "/", regex=False)
                                         .str.replace("\\", "/", regex=False))

        if labels.isnull().any().any():
            raise ValueError(
                f"Error: Plik z etykietami zawiera puste wartości po czyszczeniu!"
                f"Ścieżka: {self.labels_path}"
            )

        labels['reference_image_name'] = (
            labels['distorted_image_name']
            .apply(self._extract_reference_image_name)
        )

        return labels


    def _get_label(self, index: int) -> Label:
        label = self.labels.iloc[index]

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



class Tid2008Dataset(TidDataset): pass


class Tid2013Dataset(TidDataset): pass
