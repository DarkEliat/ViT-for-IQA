from pathlib import Path
from typing import get_args

from src.datasets.dataset_map import DATASET_LIST
from src.utils.data_types import SplitName, DatasetName
from src.utils.paths import EXPERIMENTS_PATH, CONFIGS_PATH


def validate_experiment_path_arg(experiment_path: Path) -> None:
    if not experiment_path.exists() or not experiment_path.is_dir():
        raise FileNotFoundError(
            f"Error: Eksperyment nie istnieje!\n"
            f"Ścieżka: {experiment_path}"
        )


def validate_checkpoint_path_arg(checkpoint_path: Path) -> None:
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Error: Checkpoint nie istnieje!\n"
            f"Ścieżka: {checkpoint_path}"
        )


def validate_training_config_name_arg(training_config_name: str) -> None:
    training_config_path = (CONFIGS_PATH / training_config_name).expanduser().resolve()

    if not training_config_path.exists() or not training_config_path.is_file():
        raise FileNotFoundError(
            f"Error: Treningowy plik konfiguracyjny YAML nie istnieje!\n"
            f"Ścieżka: {training_config_path}"
        )


def validate_dataset_name_arg(dataset_name: DatasetName) -> None:
    if dataset_name not in DATASET_LIST:
        raise ValueError(
            f"Error: Dataset o nazwie `{dataset_name}` nie jest dostępny!\n"
            f"Spróbuj ponownie z jednym z poniższych:\n"
            f"{DATASET_LIST}"
        )


def validate_split_name_arg(split_name: SplitName) -> None:
    available_split_names = list(get_args(SplitName))

    if split_name not in available_split_names:
        raise ValueError(
            f"Error: Split o nazwie `{split_name}` nie jest dostępny!\n"
            f"Spróbuj ponownie z jednym z poniższych:\n"
            f"{available_split_names}"
        )
