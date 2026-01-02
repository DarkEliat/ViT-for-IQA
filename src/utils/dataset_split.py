import random
import csv
from pathlib import Path

import pandas as pd


def generate_and_save_dataset_split(
        dataset_length: int,
        train_split: float,
        output_directory: Path,
        random_seed: int = 42
) -> None:
    if not output_directory.exists():
        raise FileNotFoundError(f'Error: Wskazany folder do zapisania splitów treningowego i walidacyjnego nie istnieje!\n'
                                f'Ścieżka: {output_directory}')

    all_indices = list(range(dataset_length))

    random.seed(random_seed)
    random.shuffle(all_indices)

    train_size = int(dataset_length * train_split)

    train_indices = all_indices[:train_size]
    validation_indices = all_indices[train_size:]

    save_split_indices(train_indices, (output_directory / 'train_indices.csv'))
    save_split_indices(validation_indices, (output_directory / 'validation_indices.csv'))


def load_split_indices(file_path: Path) -> list[int]:
    data_frame = pd.read_csv(file_path, header=None, dtype=int)

    indices: list[int] = data_frame.iloc[:, 0].tolist()

    return indices


def save_split_indices(indices: list[int], output_path: Path) -> None:
    if not output_path.parent.exists():
        raise FileNotFoundError('Error: Wskazany folder do zapisu indeksów datasetu nie istnieje!')

    data_frame = pd.DataFrame(indices)
    data_frame.to_csv(output_path, header=False, index=False)
