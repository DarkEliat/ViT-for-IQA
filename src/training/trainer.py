from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim import Adam, Optimizer
from torch.utils.tensorboard import SummaryWriter

from src.datasets.kadid10k_dataset import Kadid10kDataset
from src.datasets.live_dataset import LiveDataset
from src.datasets.tid2008_dataset import Tid2008Dataset
from src.datasets.tid2013_dataset import Tid2013Dataset
from src.models.vit_regressor import VitRegressor
from src.utils.config_consistency import check_consistency
from src.utils.dataset_split import load_split_indices


class Trainer:
    def __init__(self, experiment_path: Path) -> None:
        if not experiment_path.exists():
            raise FileNotFoundError('Error: Wskazany eksperyment nie istnieje!')

        self.experiment_name = experiment_path.name

        self.experiment_path = experiment_path
        self.checkpoints_path = experiment_path / 'checkpoints/'
        self.logs_path = experiment_path / 'logs/'
        self.logs_tensorboard_path = self.logs_path / 'tensorboard/'
        self.splits_path = experiment_path / 'splits/'

        config_file_path = experiment_path / 'config.yaml'
        config = check_consistency(config_file_path=config_file_path)
        self.config = config

        self.device = config['training']['device']

        # Inicjalizacja modelu
        self.model = VitRegressor(
            model_name=config['model']['name'],
            embedding_dimension=config['model']['embedding_dimension']
        ).to(self.device)

        # Definicja funkcji straty i optymalizatora
        self.loss_function: nn.Module = nn.MSELoss()
        self.optimizer: Optimizer = Adam(
            params=self.model.parameters(),
            lr=config['training']['learning_rate']
        )

        # Dataloadery
        self.train_loader, self.validation_loader = self._create_dataloaders()

        # Logowanie
        self.log_writer = (
            SummaryWriter(str(self.logs_tensorboard_path))
            if config['logging']['tensorboard']
            else None
        )


    def _build_dataset(self) -> Dataset:
        config = self.config

        match config['dataset']['name']:
            case 'kadid10k':
                return Kadid10kDataset(config=config)
            case 'tid2008':
                return Tid2008Dataset(config=config)
            case 'tid2013':
                return Tid2013Dataset(config=config)
            case 'live':
                return LiveDataset(config=config)
            case _:
                raise ValueError(f"Error: Niewspierany typ bazy danych: {config['dataset']['name']}!")


    def _create_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """
        Tworzy dataloadery do treningu i walidacji dla wybranej bazy danych.
        """

        train_indices_path = self.splits_path / 'train_indices.csv'
        validation_indices_path = self.splits_path / 'validation_indices.csv'

        if not train_indices_path.exists() or not validation_indices_path.exists():
            raise FileNotFoundError(
                f"Error: Brak co najmniej jednego pliku .csv splitu datasetu!\n"
                f"Wymagane pliki to:\n"
                f"{train_indices_path}\n"
                f"{validation_indices_path}\n"
                f"Upewnij się, że eksperyment został poprawnie utworzony."
            )

        config = self.config


        train_indices = load_split_indices(file_path=train_indices_path)
        validation_indices = load_split_indices(file_path=validation_indices_path)

        dataset: Dataset = self._build_dataset()
        train_dataset = Subset(dataset, train_indices)
        validation_dataset = Subset(dataset, validation_indices)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_of_workers']
        )

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=config['training']['batch_size'],
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
            reference_image = reference_image.to(self.device)
            distorted_image = distorted_image.to(self.device)
            dmos_value = dmos_value.to(self.device)

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
                reference_image = reference_image.to(self.device)
                distorted_image = distorted_image.to(self.device)
                dmos_value = dmos_value.to(self.device)

                prediction: Tensor = self.model(reference_image, distorted_image)
                loss: Tensor = self.loss_function(prediction, dmos_value)

                running_loss += loss.item()
                batch_count += 1

        return running_loss / max(batch_count, 1)


    def is_checkpoint_save_due(self, num_of_epoch) -> bool:
        if (self.config['checkpointing']['save_every_n_epochs'] > 0 and
            num_of_epoch % self.config['checkpointing']['save_every_n_epochs'] == 0):
            return True

        # TODO: Dodać zapisywanie ostatniej epoki i usuwanie przed ostatniej epoki,
        #  jeśli nie spełnia warunków częstotliwości ani najlepszego aktualnego wyniku
        if self.config['checkpointing']['save_last_epoch']:
            ...

        # TODO: Dodać zapisywanie najlepszej epoki, a co za tym idzie również możliwość ich porównywania
        if self.config['checkpointing']['save_best_epoch']:
            ...

        return False


    def _save_checkpoint(self, num_of_epoch: int) -> None:
        """
        Zapisuje wagi modelu do pliku.
        """

        checkpoint_path = self.checkpoints_path / f"epoch_{num_of_epoch}.pth"

        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"[Trainer] Zapisano checkpoint do: {checkpoint_path}")


    def train(self) -> None:
        num_of_epochs = self.config['training']['num_of_epochs']

        print(f"[Trainer] Startowanie treningu dla {num_of_epochs} epok...")

        for epoch in range(1, num_of_epochs+1):
            train_loss = self._train_one_epoch()
            validation_loss = self._validate_one_epoch()

            if self.log_writer:
                self.log_writer.add_scalar('loss/train', train_loss, epoch)
                self.log_writer.add_scalar('loss/validation', validation_loss, epoch)

            print(
                f"Epoka {epoch} / {num_of_epochs}  "
                f"|  Błąd trenowania: {train_loss:.4f}  "
                f"|  Błąd walidacji: {validation_loss:.4f}"
            )

            if self.is_checkpoint_save_due(num_of_epoch=epoch):
                self._save_checkpoint(epoch)
