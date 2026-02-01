from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.models.vit_regressor import VitRegressor
from src.utils.checkpoints import load_checkpoint_pickle
from src.utils.configs import load_config
from src.datasets.factory import build_data_loader
from src.utils.data_types import SplitName


class Predictor:
    def __init__(
            self,
            experiment_path: Path,
            checkpoint_name: str = 'last.pth',
            batch_size_override: int | None = None,
            num_of_workers_override: int | None = None
    ) -> None:
        if not experiment_path.exists() or not experiment_path.is_dir():
            raise FileNotFoundError(
                f"Error: Wskazany eksperyment nie istnieje!\n"
                f"Ścieżka: {experiment_path}"
            )

        self.experiment_path = experiment_path

        config_path = experiment_path / 'config.yaml'
        checkpoint_path = experiment_path / 'checkpoints' / checkpoint_name
        self.checkpoint_path = checkpoint_path

        config = load_config(config_path=config_path, check_consistency=True)

        config['training']['batch_size'] = (
            batch_size_override
            if batch_size_override
            else config['training']['batch_size']
        )
        config['training']['num_of_workers'] = (
            num_of_workers_override
            if num_of_workers_override
            else config['training']['batch_size']
        )

        self.config = config

        print(
            f"\n[Predictor] Rozpoczęto ładowanie checkpointu:\n"
            f"    Ścieżka eksperymentu: {experiment_path}\n"
            f"    Nazwa checkpointu: `{checkpoint_name}`\n"
            f"    Nazwa configu: `{self.config['config_name']}`"
        )

        self.device = config['training']['device']

        self.model = VitRegressor(
            model_name=config['model']['name'],
            embedding_dimension=config['model']['embedding_dimension']
        ).to(self.device)

        checkpoint_pickle = load_checkpoint_pickle(
            checkpoint_path=checkpoint_path,
            device=self.device,
            check_consistency=True
        )

        self.model.load_state_dict(checkpoint_pickle.model_state_dict)
        self.model.to(self.device)
        self.model.eval()

        print('\n[Predictor] Załadowano checkpoint!')


    @torch.no_grad()
    def predict(self, data_loader: DataLoader) -> list[float]:
        self.model.eval()

        predicted_quality_scores: list[float] = []

        for batch in data_loader:
            # Wsparcie obu formatów:
            # - batch jako tuple/list
            # - 2 pierwsze elementy to obrazek referencyjny i obrazek zniekształcony
            reference_image_tensor = batch[0].to(self.device)
            distorted_image_tensor = batch[1].to(self.device)

            # Forward pass: model should output shape [batch] or [batch, 1]
            model_output_tensor = self.model(reference_image_tensor, distorted_image_tensor)

            # Użycie `view(-1)` zamiast `squeeze()`:
            # - `squeeze()` może zmienić tensor w skalar, gdy batch_size=1, co komplikuje działanie `.tolist()`
            # - `view(-1)` zwraca wektor 1D
            predicted_batch_values: list[float] = (
                model_output_tensor
                .view(-1)
                .detach()
                .cpu()
                .tolist()
            )

            predicted_quality_scores.extend(predicted_batch_values)

        return predicted_quality_scores

    @torch.no_grad()
    def predict_with_ground_truth(self, data_loader: DataLoader) -> tuple[list[float], list[float]]:
        self.model.eval()

        ground_truth_scores: list[float] = []
        predicted_scores: list[float] = []

        for reference_image, distorted_image, ground_truth_score in data_loader:
            reference_image = reference_image.to(self.device)
            distorted_image = distorted_image.to(self.device)
            ground_truth_score = ground_truth_score.to(self.device)

            model_output_tensor = self.model(reference_image, distorted_image)

            predicted_batch_values: list[float] = (
                model_output_tensor
                .view(-1)
                .detach()
                .cpu()
                .tolist()
            )

            ground_truth_batch_values: list[float] = (
                ground_truth_score
                .view(-1)
                .detach()
                .cpu()
                .tolist()
            )

            predicted_scores.extend(predicted_batch_values)
            ground_truth_scores.extend(ground_truth_batch_values)

        return ground_truth_scores, predicted_scores


    @torch.no_grad()
    def predict_on_training_dataset(self, split_name: SplitName = 'full') -> list[float]:
        print('\n[Predictor] Rozpoczęto predykcję na całym datasetcie, na którym szkolony był model...')

        training_data_loader = build_data_loader(
            config=self.config,
            split_name=split_name,
            experiment_path=self.experiment_path,
        )

        predicted_quality_scores = self.predict(data_loader=training_data_loader)

        return predicted_quality_scores
