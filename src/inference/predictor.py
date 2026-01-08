from pathlib import Path
from pprint import pprint

import torch
from torch.utils.data import DataLoader, Dataset

from src.models.vit_regressor import VitRegressor
from src.utils.checkpoints import load_checkpoint_pickle
from src.utils.configs import load_config
from src.utils.data_types import Config
from src.utils.datasets import build_dataset


class Predictor:
    def __init__(
            self,
            training_config_path: Path,
            checkpoint_path: Path,
    ) -> None:
        config = load_config(config_path=training_config_path, check_consistency=True)
        self.config = config

        self.training_dataset_name = config['dataset']['name']

        self.device = config['training']['device']

        self.model = VitRegressor(
            model_name=config['model']['name'],
            embedding_dimension=config['model']['embedding_dimension']
        ).to(self.device)

        checkpoint = load_checkpoint_pickle(
            checkpoint_path=checkpoint_path,
            device=self.device,
            check_consistency=True
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(
            f"[Predictor] Załadowano checkpoint:\n"
            f"    Ścieżka checkpointu: {checkpoint_path}\n"
            f"    Nazwa configu dla tego checkpointu: `{self.config['config_name']}`"
        )

        self.training_data_loader = _build_dataloader(config=self.config)


    @torch.no_grad()
    def predict_on_training_dataset(self) -> list[float]:
        print('\n[Predictor] Rozpoczęto predykcję na całym datasetcie, na którym szkolony był model...')

        predicted_quality_scores: list[float] = []

        for reference_image_tensor, distorted_image_tensor, _ in self.training_data_loader:
            reference_image_tensor = reference_image_tensor.to(self.device)
            distorted_image_tensor = distorted_image_tensor.to(self.device)

            model_output = self.model(reference_image_tensor, distorted_image_tensor)

            predicted_quality_scores.extend(model_output.squeeze().tolist())

        return predicted_quality_scores



def _build_dataloader(config: Config) -> DataLoader:
    if (not 'dataset' in config or
        not 'name' in config['dataset']):
        raise KeyError(
            'Error: Przekazana konfiguracja nie zawiera informacji o nazwie datasetu!\n'
            'Sprawdź spójność wcześniej załadowanego pliku konfiguracyjnego YAML...'
        )

    print(
        f"[Predictor] Ładowanie datasetu `{config['dataset']['name']}` w celu wykonywania predykcji...\n"
        f"    To może potrwać nawet do kilku minut...\n"
    )

    dataset: Dataset = build_dataset(config=config)

    data_loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_of_workers']
    )

    return data_loader
