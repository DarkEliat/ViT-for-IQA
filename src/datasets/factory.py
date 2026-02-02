from pathlib import Path
from typing import get_args

from torch.utils.data import Dataset as TorchDataset, DataLoader, Subset

from src.datasets.base_dataset import BaseDataset
from src.datasets.kadid_dataset import Kadid10kDataset
from src.datasets.live_dataset import LiveDataset
from src.datasets.splits import load_split_indices
from src.datasets.tid_dataset import Tid2008Dataset, Tid2013Dataset
from src.utils.data_types import SplitName, DatasetConfig, ModelConfig, TrainingConfig


def build_dataset(
        dataset_config: DatasetConfig,
        model_config: ModelConfig
) -> TorchDataset | BaseDataset:
    match dataset_config['dataset']['name']:
        case 'kadid10k':
            return Kadid10kDataset(dataset_config=dataset_config, model_config=model_config)
        case 'tid2008':
            return Tid2008Dataset(dataset_config=dataset_config, model_config=model_config)
        case 'tid2013':
            return Tid2013Dataset(dataset_config=dataset_config, model_config=model_config)
        case 'live':
            return LiveDataset(dataset_config=dataset_config, model_config=model_config)
        case _:
            raise ValueError(f"Error: Niewspierany typ bazy danych: `{dataset_config['dataset']['name']}`!")


def build_full_data_loader(
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        shuffle: bool = False
) -> DataLoader:
    print(f"\nŁadowanie całego datasetu `{dataset_config['dataset']['name']}`...\n"
          f"    To może potrwać nawet do kilku minut...")

    dataset: TorchDataset = build_dataset(
        dataset_config=dataset_config,
        model_config=model_config
    )

    data_loader = DataLoader(
        dataset,
        batch_size=training_config['training']['batch_size'],
        shuffle=shuffle,
        num_workers=training_config['training']['num_of_workers']
    )

    return data_loader


def build_split_data_loader(
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        split_name: SplitName,
        experiment_path: Path,
        shuffle: bool = False
) -> DataLoader:
    splits_path = experiment_path / 'splits/'

    if not splits_path.exists() or not splits_path.is_dir():
        raise FileNotFoundError(
            f"Error: Folder `splits/` nie istnieje we wskazanym eksperymencie!\n"
            f"Ścieżka: {splits_path}"
        )

    split_indices_path = splits_path / f"{split_name}_indices.csv"
    if not split_indices_path.exists():
        raise FileNotFoundError(
            f"Error: Nie znaleziono pliku indeksów dla wskazanego splitu!\n"
            f"Split: `{split_name}`\n"
            f"Ścieżka: {split_indices_path}"
        )

    split_indices = load_split_indices(file_path=split_indices_path)
    if len(split_indices) == 0:
        raise ValueError(
            f"Error: Split ma pustą listę indeksów!\n"
            f"Split: `{split_name}`\n"
            f"Ścieżka: {split_indices_path}"
        )

    print(
        f"\nŁadowanie splitu `{split_name}`...\n"
        f"    Ścieżka: {split_indices_path}\n"
        f"    To może potrwać nawet do kilku minut..."
    )

    # Budowanie pełnego datasetu, a następnie ograniczenie go do `Subset`.
    dataset: TorchDataset = build_dataset(
        dataset_config=dataset_config,
        model_config=model_config
    )
    subset_dataset: Subset = Subset(dataset, split_indices)

    data_loader = DataLoader(
        subset_dataset,
        batch_size=training_config['training']['batch_size'],
        shuffle=shuffle,
        num_workers=training_config['training']['num_of_workers']
    )

    return data_loader


def build_data_loader(
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        split_name: SplitName,
        experiment_path: Path | None,
        shuffle: bool = False
) -> DataLoader:
    available_split_names = list(get_args(SplitName))
    if split_name not in available_split_names:
        raise ValueError(
            f"Error: Split o nazwie `{split_name}` nie jest dostępny!\n"
            f"Spróbuj ponownie z jednym z poniższych:\n"
            f"{available_split_names}"
        )

    if split_name != 'full' and experiment_path is None:
        raise ValueError(f"Error: Przekazano nazwę splitu bez ścieżki do eksperymentu, z którego można odczytać indeksy obrazów (*_indices.csv)!")

    if split_name == 'full':
        return build_full_data_loader(
            dataset_config=dataset_config,
            model_config=model_config,
            training_config=training_config,
            shuffle=shuffle
        )
    else:
        return build_split_data_loader(
            dataset_config=dataset_config,
            model_config=model_config,
            training_config=training_config,
            split_name=split_name,
            experiment_path=experiment_path,
            shuffle=shuffle
        )
