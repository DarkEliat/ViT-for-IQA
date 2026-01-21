import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.datasets.base_dataset import BaseDataset


def generate_split(
        dataset: BaseDataset,
        train_split: float,
        validation_split: float,
        test_split: float,
        random_seed: int,
        output_directory: Path
) -> None:
    if not output_directory.exists():
        raise FileNotFoundError(
            f"Error: Wskazany folder do zapisania splitów nie istnieje!\n"
            f"Ścieżka: {output_directory}"
        )

    # Sumy liczb zmiennoprzecinkowych mogą posiadać nieprzewidziane bardzo małe dziesiętne ułamki...
    # Z tego powodu użyta jest tolerancja zamiast równości
    split_sum = train_split + validation_split + test_split
    if abs(split_sum - 1.0) > 1e-9:
        raise ValueError(
            f"Error: Parametry `train_split`, `validation_split` oraz `test_split` muszą sumować się do 1.0 (100%) !\n"
            f"Aktualna suma: {split_sum}"
        )

    if train_split <= 0.0 or validation_split < 0.0 or test_split <= 0.0:
        raise ValueError('Error: `train_split`, oraz `test_split` muszą być > 0.0, a `validation_split` musi być >= 0.0 !')

    # Pobranie ID referencji dla każdego indeksu zniekształconego obrazu
    reference_image_names_per_distorted_image = dataset.get_reference_image_names_per_distorted_image()

    if len(reference_image_names_per_distorted_image) == 0:
        raise ValueError(
            'Error: Dataset zwrócił pustą listę obrazów referencyjnych do obrazów zniekształconych (1:1)!\n'
            'Dataset nie posiada żadnego zniekształconego obrazu albo został źle załadowany!'
        )

    dataset_length = len(dataset)

    # Grupowanie zniekształconych obrazów po ID ich obrazów referencyjnych
    reference_name_to_distorted_index_map: dict[str, list[int]] = defaultdict(list)

    for distorted_index, reference_name in enumerate(reference_image_names_per_distorted_image):
        reference_name_to_distorted_index_map[reference_name].append(distorted_index)

    all_reference_image_names = list(reference_name_to_distorted_index_map.keys())
    num_of_reference_images = len(all_reference_image_names)

    if num_of_reference_images < 2:
        raise ValueError(
            'Error: W datasetcie jest mniej niż 2 obrazy referencyjne. '
            'Nie da się poprawnie zrobić splitu train / validation / test po referencjach.'
        )

    # Deterministyczne pomieszanie nazw obrazów referencyjnych i wykonanie ich splitów
    random_generator = random.Random(random_seed)
    random_generator.shuffle(all_reference_image_names)

    train_reference_count = int(num_of_reference_images * train_split)
    validation_reference_count = int(num_of_reference_images * validation_split)
    test_reference_count = num_of_reference_images - train_reference_count - validation_reference_count

    # Upewnienie się, czy train split i test split posiadają, chociaż 1 zdjęcie referencyjne
    # To będzie miało znaczenie dla bardzo małych datasetów.
    if train_reference_count == 0:
        train_reference_count = 1
        test_reference_count = num_of_reference_images - train_reference_count - validation_reference_count

    if test_reference_count == 0:
        test_reference_count = 1

        # Redukcja train split (o ile to możliwe), w przeciwnym wypadku redukcja validation split
        if train_reference_count > 1:
            train_reference_count -= 1
        elif validation_reference_count > 0:
            validation_reference_count -= 1
        else:
            raise ValueError(f"Error: Nie da się zapewnić niepustych splitów train i test przy aktualnych proporcjach i liczbie referencji = {num_of_reference_images} !")

    train_reference_names = set(all_reference_image_names[:train_reference_count])

    validation_reference_start = train_reference_count
    validation_reference_end = train_reference_count + validation_reference_count
    validation_reference_names = set(all_reference_image_names[validation_reference_start : validation_reference_end])

    test_reference_names = set(all_reference_image_names[validation_reference_end :])

    # Dokonanie podziału obrazów zniekształconych na splity na podstawie podziału obrazów referencyjnych
    train_distorted_indices: list[int] = []
    validation_distorted_indices: list[int] = []
    test_distorted_indices: list[int] = []

    for reference_name, distorted_indices in reference_name_to_distorted_index_map.items():
        if reference_name in train_reference_names:
            train_distorted_indices.extend(distorted_indices)
        elif reference_name in validation_reference_names:
            validation_distorted_indices.extend(distorted_indices)
        else:
            test_distorted_indices.extend(distorted_indices)

    train_distorted_indices.sort()
    validation_distorted_indices.sort()
    test_distorted_indices.sort()

    all_split_indices = train_distorted_indices + validation_distorted_indices + test_distorted_indices
    if len(all_split_indices) != dataset_length:
        raise RuntimeError(
            f"Error: Suma indeksów w splitach nie pokrywa się z długością datasetu.\n"
            f"    `dataset_length`={dataset_length}\n"
            f"    `split_total`={len(all_split_indices)}"
        )

    if len(set(all_split_indices)) != dataset_length:
        raise RuntimeError('Error: Wykryto duplikaty indeksów między splitami (overlap)! To oznacza błąd w logice tworzenia splitów!')

    save_split_indices(train_distorted_indices, (output_directory / "train_indices.csv"))
    save_split_indices(validation_distorted_indices, (output_directory / "validation_indices.csv"))
    save_split_indices(test_distorted_indices, (output_directory / "test_indices.csv"))


def load_split_indices(file_path: Path) -> list[int]:
    data_frame = pd.read_csv(file_path, header=None, dtype=int)

    indices: list[int] = data_frame.iloc[:, 0].tolist()

    return indices


def save_split_indices(indices: list[int], output_path: Path) -> None:
    if not output_path.parent.exists():
        raise FileNotFoundError('Error: Wskazany folder do zapisu indeksów datasetu nie istnieje!')

    data_frame = pd.DataFrame(indices)
    data_frame.to_csv(output_path, header=False, index=False)