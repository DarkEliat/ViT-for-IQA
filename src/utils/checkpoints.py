from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from src.utils.data_types import CheckpointPickle


def load_checkpoint_pickle(
        checkpoint_path: Path, device: str,
        check_consistency: bool
) -> CheckpointPickle:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Error: Nie znaleziono wskazanego checkpointu!\n"
            f"Ścieżka: {checkpoint_path}"
        )

    checkpoint_dict = torch.load(checkpoint_path, map_location=device)

    if check_consistency:
        _check_checkpoint_dict_consistency(
            checkpoint_dict=checkpoint_dict,
            path=checkpoint_path
        )

    checkpoint_pickle = CheckpointPickle(
        model_state_dict=checkpoint_dict['model_state_dict'],
        optimizer_state_dict=checkpoint_dict['optimizer_state_dict'],
        last_epoch=checkpoint_dict['last_epoch'],
        best_epoch=checkpoint_dict['best_epoch'],
        config=checkpoint_dict['config']
    )

    return checkpoint_pickle


def _check_checkpoint_dict_consistency(
        checkpoint_dict: dict[str, Any],
        path: Path | None = None
) -> None:
    checkpoint_pickle_dataclass_fields = fields(CheckpointPickle)

    required_keys = {
        field.name
        for field in checkpoint_pickle_dataclass_fields
    }

    missing_keys = required_keys - checkpoint_dict.keys()
    if missing_keys:
        error_message = (
            f"Error: Checkpoint jest niekompletny!\n"
            f"Brakujące klucze w pliku .pth: {sorted(missing_keys)}\n"
            f"Ścieżka: {path if path else 'NIEZNANA'}"
        )

        print(error_message)
        raise KeyError(error_message)
