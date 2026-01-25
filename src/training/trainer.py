from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim import Adam, Optimizer
from torch.utils.tensorboard import SummaryWriter

from src.models.vit_regressor import VitRegressor
from src.utils.checkpoints import load_checkpoint_pickle
from src.utils.configs import load_config
from src.datasets.factory import build_dataset
from src.datasets.splits import load_split_indices


class Trainer:
    def __init__(self, experiment_path: Path) -> None:
        if not experiment_path.exists():
            raise FileNotFoundError(
                f"Error: Wskazany eksperyment nie istnieje!"
                f"Ścieżka: {experiment_path}"
            )

        self.experiment_name = experiment_path.name

        self.experiment_path = experiment_path
        self.checkpoints_path = experiment_path / 'checkpoints/'
        self.logs_path = experiment_path / 'logs/'
        self.logs_tensorboard_path = self.logs_path / 'tensorboard/'
        self.splits_path = experiment_path / 'splits/'

        config_path = experiment_path / 'config.yaml'
        config = load_config(config_path=config_path, check_consistency=True)
        self.config = config

        self.dataset_name = config['dataset']['name']

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

        # Logowanie
        self.log_writer = (
            SummaryWriter(str(self.logs_tensorboard_path))
            if config['logging']['tensorboard']
            else None
        )

        print(
            f"[Trainer] Załadowano eksperyment:\n"
            f"    Nazwa eksperymentu: `{self.experiment_name}`\n"
            f"    Ścieżka eksperymentu: {self.experiment_path}\n"
            f"    Nazwa configu: `{self.config['config_name']}`"
        )

        # Data Loadery
        self.train_loader, self.validation_loader = self._build_data_loaders()


    def _build_data_loaders(self) -> tuple[DataLoader, DataLoader]:
        """
        Tworzy data loadery do treningu i walidacji dla wybranej bazy danych.
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

        print(
            f"\n[Trainer] Ładowanie datasetu `{self.dataset_name}` w celu szkolenia modelu ViT...\n"
            f"    To może potrwać nawet do kilku minut..."
        )

        config = self.config

        train_indices = load_split_indices(file_path=train_indices_path)
        validation_indices = load_split_indices(file_path=validation_indices_path)

        dataset: Dataset = build_dataset(config=config)
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


    def train_one_epoch(self) -> float:
        """
        Wykonuje jedną epokę szkolenia.
        """

        self.model.train()

        running_loss = 0.0
        batch_count = 0

        for reference_image_tensor, distorted_image_tensor, dmos_value_tensor in self.train_loader:
            reference_image_tensor = reference_image_tensor.to(self.device)
            distorted_image_tensor = distorted_image_tensor.to(self.device)
            dmos_value_tensor = dmos_value_tensor.to(self.device)

            self.optimizer.zero_grad()

            prediction: Tensor = self.model(reference_image_tensor, distorted_image_tensor)
            loss: Tensor =  self.loss_function(prediction, dmos_value_tensor)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            batch_count += 1

        return running_loss / max(batch_count, 1)


    def validate_last_epoch(self) -> float:
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


    def check_checkpoint_save_due(self, epoch) -> dict[str, bool]:
        save_checkpoint_triggers: dict[str, bool] = {
            'save_every_n_epochs': False,
            'save_last_epoch': False,
            'save_best_epoch': False
        }

        if (self.config['checkpointing']['save_every_n_epochs'] > 0 and
            epoch % self.config['checkpointing']['save_every_n_epochs'] == 0):
            save_checkpoint_triggers['save_every_n_epochs'] = True

        if self.config['checkpointing']['save_last_epoch']:
            save_checkpoint_triggers['save_last_epoch'] = True

        # TODO: Dodać zapisywanie najlepszej epoki, a co za tym idzie również możliwość ich porównywania
        if self.config['checkpointing']['save_best_epoch']:
            ...

        return save_checkpoint_triggers


    def save_checkpoint(
            self,
            epoch: int,
            train_loss: float,
            validation_loss: float,
            save_triggers: dict[str, bool]
    ) -> None:
        """
        Zapisuje wagi modelu do pliku.
        """

        if all(value is False for value in save_triggers.values()):
            print('WARNING: [Trainer] Nie przekazano żadnego powodu wykonania checkpointu!\n'
                  '    SPRAWDŹ POWÓD TEJ SYTUACJI! Checkpointy mogą być potencjalnie źle zapisywane!'
                  '    Aktualnie ustawiono domyślny powód: `save_last_epoch`.')

            save_triggers['save_last_epoch'] = True

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'validation_loss': validation_loss
        }

        print(f"[Trainer] Zapisano checkpoint do:")

        if save_triggers['save_every_n_epochs']:
            epoch_checkpoint_path = self.checkpoints_path / f"epoch_{epoch}.pth"
            torch.save(checkpoint, epoch_checkpoint_path)

            print(f"    {epoch_checkpoint_path}")

        if save_triggers['save_last_epoch']:
            last_checkpoint_path = self.checkpoints_path / 'last.pth'
            torch.save(checkpoint, last_checkpoint_path)

            print(f"    {last_checkpoint_path}")

        if save_triggers['save_best_epoch']:
            best_checkpoint_path = self.checkpoints_path / 'best.pth'
            torch.save(checkpoint, best_checkpoint_path)

            print(f"    {best_checkpoint_path}")


    def load_checkpoint(self, checkpoint_path: Path) -> int:
        checkpoint = load_checkpoint_pickle(
            checkpoint_path=checkpoint_path,
            device=self.device,
            check_consistency=True
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch'] + 1

        print(f"\n[Trainer] Wczytano checkpoint z {checkpoint['epoch']}. epoki (kontynuacja od {start_epoch}. epoki).")

        return start_epoch


    def any_checkpoint_exists(self):
        if not self.checkpoints_path.exists() or not self.checkpoints_path.is_dir():
            raise FileNotFoundError(
                f"Error: Nie istnieje folder checkpointów"
                f"Ścieżka: {self.checkpoints_path}"
            )

        return any(self.checkpoints_path.glob('*.pth'))


    def try_resume(self) -> int:
        if not self.any_checkpoint_exists():
            return 0

        last_checkpoint_exist = any(self.checkpoints_path.glob('last.pth'))
        init_checkpoint_exist = any(self.checkpoints_path.glob('init.pth'))
        best_checkpoint_exist = any(self.checkpoints_path.glob('best.pth'))
        epoch_checkpoint_exist = any(self.checkpoints_path.glob('epoch_*.pth'))

        if not last_checkpoint_exist and not init_checkpoint_exist:
            return 0

        checkpoint_to_load = self.checkpoints_path

        if last_checkpoint_exist and not self.config['checkpointing']['save_last_epoch']:
            print(
                '\nWARNING: [Trainer] Znaleziono checkpoint `last.pth`.\n'
                '    Jednocześnie parametr konfiguracyjny `checkpointing.save_last_epoch` jest ustawiony na `false`.\n'
                '    Oznacza to, że przy aktualnej konfiguracji wskazany checkpoint nie mógł powstać.\n'
                '    Jeśli celowo umieściłeś plik `last.pth` w folderze `checkpoints\n'
                '    i jesteś pewien, że chcesz kontynuować trening, to wpisz [y / Y / t / T].\n'
                '    Jeśli nie chcesz kontynuować, to przerwij działanie programu albo wpisz [n / N].'
            )

            checkpoint_to_load /= 'last.pth'

        elif last_checkpoint_exist:
            print(
                '\nWARNING: [Trainer] Znaleziono checkpoint `last.pth`.\n'
                '    Jeśli jesteś pewien, że chcesz kontynuować trening, to wpisz [y / Y / t / T].\n'
                '    Jeśli nie chcesz kontynuować, to przerwij działanie programu albo wpisz [n / N].'
            )

            checkpoint_to_load /= 'last.pth'

        elif best_checkpoint_exist or epoch_checkpoint_exist:
            print(
                '\nWARNING: [Trainer] Znaleziono checkpoint(y) `epoch_*.pth` lub `best.pth`.\n'
                '    Jednocześnie nie znaleziono checkpointu `last.pth`, który umożliwiałby bezpieczne kontynuowanie treningu.'
            )

            input('Aby zapobiec utracie dotychczasowych wyników pracy w eksperymencie PRZERYWAM DZIAŁANIE PROGRAMU!\n'
                  'Naciśnij ENTER...')
            exit(1)

        elif init_checkpoint_exist:
            print(
                '\nWARNING: [Trainer] Znaleziono checkpoint `init.pth`.\n'
                '    Jeśli stworzyłeś cały ten eksperyment na podstawie checkpointu z innego eksperymentu\n'
                '    albo celowo umieściłeś plik `init.pth` w folderze `checkpoints/ jako pretrenowany model`\n'
                '    i jesteś pewien, że chcesz kontynuować trening, to wpisz [y / Y / t / T].\n'
                '    Jeśli nie chcesz kontynuować, to przerwij działanie programu albo wpisz [n / N].'
            )

            checkpoint_to_load /= 'init.pth'

        while checkpoint_to_load.is_file():
            try:
                choice = input('> ')
                choice = choice.lower()

                if choice in ('y', 't'):
                    return self.load_checkpoint(checkpoint_path=checkpoint_to_load)
                elif choice == 'n':
                    input('PRZERYWAM DZIAŁANIE PROGRAMU! Naciśnij Enter...')
                    exit(1)
                else:
                    raise ValueError('Wpisano niepoprawną wartość! Spróbuj jeszcze raz!')

            except ValueError as error:
                print(error)

        return 0

    def train(self, resume_from: Path | None = None) -> None:
        num_of_epochs = self.config['training']['num_of_epochs']
        start_epoch = 1

        epoch_from_checkpoint = self.try_resume()

        if epoch_from_checkpoint > 1:
            start_epoch = epoch_from_checkpoint
        else:
            print(f"\n[Trainer] Startowanie treningu dla {num_of_epochs} epok...")

        for epoch in range(start_epoch, num_of_epochs+1):
            print(f"\n[Trainer] Rozpoczęto trening {epoch}. epoki.")

            train_loss = self.train_one_epoch()
            validation_loss = self.validate_last_epoch()

            if self.log_writer:
                self.log_writer.add_scalar('loss/train', train_loss, epoch)
                self.log_writer.add_scalar('loss/validation', validation_loss, epoch)

            print(
                f"[Trainer] Ukończono trening epoki {epoch} / {num_of_epochs}    "
                f"|    Błąd trenowania: {train_loss:.4f}    "
                f"|    Błąd walidacji: {validation_loss:.4f}"
            )

            save_checkpoint_triggers = self.check_checkpoint_save_due(epoch=epoch)
            if any(save_checkpoint_triggers.values()):
                self.save_checkpoint(epoch=epoch,
                                     train_loss=train_loss,
                                     validation_loss=validation_loss,
                                     save_triggers=save_checkpoint_triggers)

        if start_epoch <= num_of_epochs:
            print(f"\n[Trainer] Cały trening zakończony!\n"
                  f"    Przeprowadzono {num_of_epochs} epok szkolenia.\n")
        else:
            print(f"\n[Trainer] Nie można przeprowadzić treningu, ponieważ numer startowej epoki ({start_epoch})\n"
                  f"    jest większy niż maksymalna liczba epok treningu ({num_of_epochs}) w pliku konfiguracyjnym YAML eksperymentu.")

        input('Naciśnij Enter, aby zakończyć...')
