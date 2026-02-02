from pathlib import Path

import yaml
from typing_extensions import get_args

from src.configs.dataset_config import check_dataset_config_consistency
from src.configs.model_config import check_model_config_consistency
from src.configs.training_config import check_training_config_consistency
from src.utils.data_types import ModelConfig, ConfigType


def load_config(config_path: Path, config_type: ConfigType, check_consistency: bool) -> ModelConfig:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Error: Plik konfiguracyjny nie istnieje!\n"
            f"Ścieżka: {config_path}"
        )

    if config_path.suffix not in {'.yaml', '.yml'}:
        raise ValueError(
            f"Error: Nieprawidłowy format pliku konfiguracyjnego! Oczekiwano pliku YAML (.yaml / .yml)!\n"
            f"Ścieżka: {config_path}"

        )

    available_config_types = list(get_args(ConfigType))
    if config_type not in available_config_types:
        raise ValueError(
            f"Error: Przekazano nieprawidłowy rodzaj pliku konfiguracyjnego YAML!\n"
            f"Dostępne rodzaju:\n"
            f"{available_config_types}"
        )

    with open(config_path, 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise TypeError(
            f"Error: Nieprawidłowa struktura pliku YAML! Oczekiwano słownika (dict) na najwyższym poziomie!\n"
            f"Ścieżka: {config_path}"
        )

    if check_consistency:
        match config_type:
            case 'dataset':
                check_dataset_config_consistency(config=config, path=config_path)
            case 'model':
                check_model_config_consistency(config=config, path=config_path)
            case 'training':
                check_training_config_consistency(config=config, path=config_path)
            case _:
                raise ValueError(f"Error: Sprawdzenie spójności pliku konfiguracyjnego YAML z rodzaju `{config_type}` nie jest jeszcze wspierane!")

    return config



