import pandas as pd

from src.datasets.base_dataset import BaseDataset
from src.utils.data_types import Label, QualityScore, UnifiedQualityScore
from src.utils.quality_scores import dmos_to_quality_score


class Kadid10kDataset(BaseDataset[ pd.DataFrame ]):
    def __init__(self, config):
        super().__init__(config=config)

        self._labels_container = self._build_labels_container()


    @property
    def labels_container(self) -> pd.DataFrame:
        return self._labels_container


    def _unify_quality_score(self, value: float) -> UnifiedQualityScore:
        quality_label_type = self.config['dataset']['quality_label']['type']
        if quality_label_type != 'dmos':
            raise TypeError(
                'Error: Nie można konwertować wskaźnika DMOS na zdefiniowane globalnie `UnifiedQualityScore`,'
                '    ponieważ plik konfiguracyjny YAML wskazuje inny typ wskaźnika jakości niż `dmos`!'
            )

        dmos_min = self.config['dataset']['quality_label']['min']
        dmos_max = self.config['dataset']['quality_label']['max']

        return dmos_to_quality_score(dmos_value=value, dmos_min=dmos_min, dmos_max=dmos_max)



    def get_reference_image_name(self, distorted_image_name: str) -> str:
        matches = self.labels_container.loc[
            self.labels_container['distorted_image_name'] == distorted_image_name,
            'reference_image_name'
        ]

        if matches.empty:
            raise KeyError(f"Error: Nie znaleziono obrazu referencyjnego dla obrazu `{distorted_image_name}`!")

        if len(matches) > 1:
            raise ValueError(f"Error: Znaleziono wiele (zamiast TYLKO jednego) obrazów referencyjnych dla obrazu `{distorted_image_name}`!")

        return matches.iloc[0]


    def get_reference_image_names_per_distorted_image(self) -> list[str]:
        return self.labels_container['reference_image_name'].astype(str).tolist()


    def _build_labels_container(self) -> pd.DataFrame:
        labels_container = pd.read_csv(
            self.labels_path,
            sep=r',',
            header=0
        )

        labels_container.columns = labels_container.columns.str.strip()

        labels_container.astype({
            'dist_img': str,
            'ref_img': str,
            'dmos': float,
            'var': float
        })

        labels_container = labels_container.rename(
            columns={
                'ref_img': 'reference_image_name',
                'dist_img': 'distorted_image_name',
                'dmos': 'quality_score'
            }
        )

        for column_name in labels_container.select_dtypes(include='object').columns:
            labels_container[column_name] = (labels_container[column_name].str.strip()
                                                      .str.replace("\\\\", "/", regex=False)
                                                      .str.replace("\\", "/", regex=False))

        if labels_container.isnull().any().any():
            raise ValueError(
                f"Error: Plik z etykietami zawiera puste wartości po czyszczeniu!"
                f"Ścieżka: {self.labels_path}"
            )

        return labels_container


    def get_label(self, index: int) -> Label:
        label = self.labels_container.iloc[index]

        return Label(
            reference_image_name=label['reference_image_name'],
            distorted_image_name=label['distorted_image_name'],
            quality_score=QualityScore(
                type='dmos',
                min_value=self.config['dataset']['quality_label']['min'],
                max_value=self.config['dataset']['quality_label']['max'],
                value=label['quality_score'],
                normalized=False,
                model_target=False
            )
        )
