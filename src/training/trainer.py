from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, random_split, Dataset
from torch.optim import Adam, Optimizer
from torch.utils.tensorboard import SummaryWriter

from src.data.kadid10k_dataset import Kadid10kDataset
from src.data.live_dataset import LiveDataset
from src.data.tid2008_dataset import Tid2008Dataset
from src.data.tid2013_dataset import Tid2013Dataset
from src.models.vit_regressor import VitRegressor


class Trainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.config['logging']['checkpoints_path'] = Path(self.config['logging']['checkpoints_path'])

        self.computation_unit = config['training']['device']

        # Logowanie
        experiment_name = config['experiment_name']
        self.writer = (
            SummaryWriter(f'logs/{experiment_name}')
            if config['logging']['tensorboard']
            else None
        )

        # Inicjalizacja modelu
        self.model = VitRegressor(
            model_name=config['model']['name'],
            embedding_dimension=config['model']['embedding_dimension']
        ).to(self.computation_unit)

        # Definicja funkcji straty i optymalizatora
        self.loss_function: nn.Module = nn.MSELoss()
        self.optimizer: Optimizer = Adam(
            params=self.model.parameters(),
            lr=config['training']['learning_rate']
        )

        # Dataloadery
        self.train_loader, self.validation_loader = self._create_dataloaders()


    def _build_dataset(self) -> Dataset:
        config = self.config

        match config['type']:
            case 'kadid10k':
                return Kadid10kDataset(
                    images_folder_path=config['dataset']['images_path'],
                    csv_file_path=config['dataset']['labels_path'],
                    transform=None
                )
            case 'tid2008':
                return Tid2008Dataset()
            case 'tid2013':
                return Tid2013Dataset()
            case 'live':
                return LiveDataset()
            case _:
                raise ValueError(f"Error: Niewspierany typ bazy danych: {config['dataset']['type']}!")


    def _create_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """
        Tworzy dataloadery do treningu i walidacji dla wybranej bazy danych.
        """

        config = self.config

        dataset: Dataset = self._build_dataset()

        whole_dataset_length = len(dataset)  # type: ignore

        train_dataset_length = int(config['dataset']['train_split'] * whole_dataset_length)
        validation_dataset_length = whole_dataset_length - train_dataset_length

        train_dataset, validation_dataset = random_split(
            dataset,
            lengths=[train_dataset_length, validation_dataset_length]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['dataset']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_of_workers']
        )

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=config['dataset']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_of_workers']
        )

        return train_loader, validation_loader


    def _train_one_epoch(self) -> float:
        """
        Wykonuje jedną epokę szkolenia.
        """

        self.model.train()

        running_loss = 0.0
        batch_count = 0

        for reference_image, distorted_image, dmos_value in self.train_loader:
            reference_image = reference_image.to(self.computation_unit)
            distorted_image = distorted_image.to(self.computation_unit)
            dmos_value = dmos_value.to(self.computation_unit)

            self.optimizer.zero_grad()

            prediction: Tensor = self.model(reference_image, distorted_image)
            loss: Tensor =  self.loss_function(prediction, dmos_value)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            batch_count += 1

        return running_loss / max(batch_count, 1)


    def _validate_one_epoch(self) -> float:
        """
        Wykonuje jedną epokę walidacji.
        """

        self.model.eval()

        running_loss = 0.0
        batch_count = 0

        with torch.no_grad():
            for reference_image, distorted_image, dmos_value in self.validation_loader:
                reference_image = reference_image.to(self.computation_unit)
                distorted_image = distorted_image.to(self.computation_unit)
                dmos_value = dmos_value.to(self.computation_unit)

                prediction: Tensor = self.model(reference_image, distorted_image)
                loss: Tensor = self.loss_function(prediction, dmos_value)

                running_loss += loss.item()
                batch_count += 1

        return running_loss / max(batch_count, 1)


    def _save_checkpoint(self, num_of_epoch: int) -> None:
        """
        Zapisuje wagi modelu do pliku.
        """

        checkpoint_path = self.config['logging']['checkpoints_path'] / f"epoch_{num_of_epoch}.pth"

        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"[Trainer] Zapisano checkpoint do: {checkpoint_path}")


    def train(self) -> None:
        num_of_epochs = self.config['training']['num_of_epochs']

        print(f"[Trainer] Startowanie treningu dla {num_of_epochs} epok...")

        for epoch in range(1, num_of_epochs+1):
            train_loss = self._train_one_epoch()
            validation_loss = self._validate_one_epoch()

            if self.writer:
                self.writer.add_scalar('loss/train', train_loss, epoch)
                self.writer.add_scalar('loss/validation', validation_loss, epoch)

            print(
                f"Epoka {epoch} / {num_of_epochs}  "
                f"|  Błąd trenowania: {train_loss:.4f}  "
                f"|  Błąd walidacji: {validation_loss:.4f}"
            )

            if epoch % self.config['logging']['save_checkpoint_every'] == 0:
                self._save_checkpoint(epoch)
