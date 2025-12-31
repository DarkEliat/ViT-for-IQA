import random
import csv
from pathlib import Path


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

    _save_split_indices(train_indices, (output_directory / 'train_indices.csv'))
    _save_split_indices(validation_indices, (output_directory / 'validation_indices.csv'))


def load_split_indices(file_path: Path) -> list[int]:
    indices = []

    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)

        for row in reader:
            indices.append(int(row[0]))

    return indices


def _save_split_indices(indices: list[int], output_path: Path) -> None:
    with open(output_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        for index in indices:
            writer.writerow([index])
