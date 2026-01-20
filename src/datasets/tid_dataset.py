import pandas as pd

from src.datasets.base_dataset import BaseDataset
from src.utils.data_types import Label, QualityScore, UnifiedQualityScore
from src.utils.quality_scores import normalize_min_max


class TidDataset(BaseDataset[ pd.DataFrame ]):
    def __init__(self, config):
        super().__init__(config=config)

        self._labels_container = self._build_labels_container()


    @property
    def labels_container(self) -> pd.DataFrame:
        return self._labels_container


    def _unify_quality_score(self, value: float) -> UnifiedQualityScore:
        quality_label_type = self.config['dataset']['quality_label']['type']
        if quality_label_type != 'mos':
            raise TypeError(
                'Error: Nie można konwertować wskaźnika MOS na zdefiniowane globalnie `UnifiedQualityScore`,'
                '    ponieważ plik konfiguracyjny YAML wskazuje inny typ wskaźnika jakości niż `mos`!'
            )

        mos_min = self.config['dataset']['quality_label']['min']
        mos_max = self.config['dataset']['quality_label']['max']

        normalized_mos = normalize_min_max(score_value=value, score_min=mos_min, score_max=mos_max)

        return UnifiedQualityScore(value=normalized_mos)


    def get_reference_image_name(self, distorted_image_name: str) -> str:
        reference_prefix = distorted_image_name.split(sep='_', maxsplit=1)[0]

        reference_image_name = f"{reference_prefix}.bmp"

        if not self.reference_images_map.has_file_name(file_name=reference_image_name):
            raise FileNotFoundError(
                f"Error: Nie znaleziono obrazu referencyjnego dla: {distorted_image_name}!\n"
                f"Ścieżka: {self.reference_images_path}"
            )

        return reference_image_name


    def get_reference_image_names_per_distorted_image(self) -> list[str]:
        return self.labels_container['reference_image_name'].astype(str).tolist()


    def _build_labels_container(self) -> pd.DataFrame:
        labels_container: pd.DataFrame = pd.read_csv(
            self.labels_path,
            sep=r'\s+',  # Dowolna liczba białych znaków jako separator kolumn
            header=None,
            names=['quality_score', 'distorted_image_name'],
            dtype={
                'quality_score': float,
                'distorted_image_name': str
            }
        )
        labels_container.columns = labels_container.columns.str.strip()

        for column_name in labels_container.select_dtypes(include='object').columns:
            labels_container[column_name] = (labels_container[column_name].str.strip()
                                         .str.replace("\\\\", "/", regex=False)
                                         .str.replace("\\", "/", regex=False))

        if labels_container.isnull().any().any():
            raise ValueError(
                f"Error: Plik z etykietami zawiera puste wartości po czyszczeniu!"
                f"Ścieżka: {self.labels_path}"
            )

        labels_container['reference_image_name'] = (
            labels_container['distorted_image_name']
            .apply(self.get_reference_image_name)
        )

        return labels_container


    def get_label(self, index: int) -> Label:
        label = self.labels_container.iloc[index]

        return Label(
            reference_image_name=label['reference_image_name'],
            distorted_image_name=label['distorted_image_name'],
            quality_score=QualityScore(
                type='mos',
                min_value=self.config['dataset']['quality_label']['min'],
                max_value=self.config['dataset']['quality_label']['max'],
                value=label['quality_score'],
                normalized=False,
                model_target=False
            )
        )



class Tid2008Dataset(TidDataset): pass


class Tid2013Dataset(TidDataset): pass
