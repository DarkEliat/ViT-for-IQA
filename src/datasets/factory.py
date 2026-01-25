from pathlib import Path

from torch.utils.data import Dataset as TorchDataset, DataLoader, Subset

from src.datasets.base_dataset import BaseDataset
from src.datasets.kadid_dataset import Kadid10kDataset
from src.datasets.live_dataset import LiveDataset
from src.datasets.splits import load_split_indices
from src.datasets.tid_dataset import Tid2008Dataset, Tid2013Dataset
from src.utils.data_types import Config, SplitName


def build_dataset(config: Config) -> TorchDataset | BaseDataset:
    match config['dataset']['name']:
        case 'kadid10k':
            return Kadid10kDataset(config=config)
        case 'tid2008':
            return Tid2008Dataset(config=config)
        case 'tid2013':
            return Tid2013Dataset(config=config)
        case 'live':
            return LiveDataset(config=config)
        case _:
            raise ValueError(f"Error: Niewspierany typ bazy danych: `{config['dataset']['name']}`!")


def build_data_loader(config: Config) -> DataLoader:
    if (
            not 'dataset' in config
            or
            not 'name' in config['dataset']
    ):
        raise KeyError(
            'Error: Przekazana konfiguracja nie zawiera informacji o nazwie datasetu!\n'
            'Sprawdź spójność wcześniej załadowanego pliku konfiguracyjnego YAML...'
        )

    print(f"\nŁadowanie datasetu `{config['dataset']['name']}`...\n"
          f"    To może potrwać nawet do kilku minut...")

    dataset: TorchDataset = build_dataset(config=config)

    data_loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_of_workers']
    )

    return data_loader


def build_split_data_loader(
        config: Config,
        split_name: SplitName,
        experiment_path: Path
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
        f"Ładowanie splitu `{split_name}`...\n"
        f"    Ścieżka: {split_indices_path}\n"
        f"    To może potrwać nawet do kilku minut..."
    )

    # Budowanie pełnego datasetu, a następnie ograniczenie go do `Subset`.
    dataset: TorchDataset = build_dataset(config=config)
    subset_dataset: Subset = Subset(dataset, split_indices)

    data_loader = DataLoader(
        subset_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_of_workers']
    )

    return data_loader
