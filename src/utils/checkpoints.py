from pathlib import Path

import torch

from src.utils.data_types import Checkpoint


def load_checkpoint_pickle(checkpoint_path: Path, device: str, check_consistency: bool) -> Checkpoint:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Error: Nie znaleziono wskazanego checkpointu!\n"
            f"Ścieżka: {checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if check_consistency:
        check_checkpoint_consistency(checkpoint=checkpoint, path=checkpoint_path)

    return checkpoint


def check_checkpoint_consistency(checkpoint: Checkpoint, path: Path | None = None):
    required_keys = {
        'epoch',
        'model_state_dict',
        'optimizer_state_dict',
        'train_loss',
        'validation_loss'
    }

    missing_keys = required_keys - checkpoint.keys()
    if missing_keys:
        raise KeyError(
            f"Error: Checkpoint jest niekompletny!\n"
            f"Brakuje: {sorted(missing_keys)}\n"
            f"Ścieżka: {path if path else 'NIEZNANA'}"
        )
