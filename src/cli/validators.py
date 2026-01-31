from pathlib import Path
from typing import get_args

from src.utils.data_types import SplitName
from src.utils.paths import EXPERIMENTS_PATH


def validate_experiment_path_arg(experiment_path: str) -> Path:
    experiment_path = (EXPERIMENTS_PATH / experiment_path).expanduser().resolve()

    if not experiment_path.exists() or not experiment_path.is_dir():
        raise FileNotFoundError(
            f"Error: Eksperyment nie istnieje!\n"
            f"Ścieżka: {experiment_path}"
        )

    return experiment_path


def validate_split_name_arg(experiment_path: Path, split_name: SplitName) -> SplitName:
    available_split_names = list(get_args(SplitName))

    if split_name not in available_split_names:
        raise ValueError(
            f"Error: Split o nazwie `{split_name}` nie jest dostępny!\n"
            f"Spróbuj ponownie z jednym z poniższych:\n"
            f"{available_split_names}"
        )

    if split_name != 'full':
        split_path = (experiment_path / 'splits' / f"{split_name}_indices.csv").resolve()

        if not split_path.exists():
            raise FileNotFoundError(
                f"Error: Wskazany split nie istnieje!\n"
                f"Ścieżka: {split_path}"
            )

    return split_name
