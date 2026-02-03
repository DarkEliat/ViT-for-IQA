from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam, Optimizer
from torch.utils.tensorboard import SummaryWriter

from src.evaluation.correlation_metrics import CorrelationMetrics, compute_correlations
from src.models.vit_regressor import VitRegressor
from src.utils.checkpoints import load_checkpoint_pickle
from src.configs.loader import load_config
from src.datasets.factory import build_data_loader
from src.utils.data_types import LossMetrics, CheckpointInfo, CheckpointPickle


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

        dataset_config_path = experiment_path / 'configs' / 'dataset.yaml'
        model_config_path = experiment_path / 'configs' / 'model.yaml'
        training_config_path = experiment_path / 'configs' / 'training.yaml'

        dataset_config = load_config(
            config_path=dataset_config_path,
            config_type='dataset',
            check_consistency=True
        )
        model_config = load_config(
            config_path=model_config_path,
            config_type='model',
            check_consistency=True
        )
        training_config = load_config(
            config_path=training_config_path,
            config_type='training',
            check_consistency=True
        )

        self.dataset_config = dataset_config
        self.model_config = model_config
        self.training_config = training_config

        print(
            f"\n[Trainer] Rozpoczęto ładowanie eksperymentu:\n"
            f"    Nazwa eksperymentu: `{self.experiment_name}`\n"
            f"    Ścieżka eksperymentu: {self.experiment_path}\n"
            f"    Nazwa configu treningowego: `{self.training_config['config_name']}`"
        )

        self.dataset_name = dataset_config['dataset']['name']

        self.device = training_config['training']['device']

        # Inicjalizacja modelu
        self.model = VitRegressor(
            model_name=model_config['model']['name'],
            embedding_dimension=model_config['model']['embedding_dimension']
        ).to(self.device)

        # Funkcja straty
        self.loss_function: nn.Module = nn.MSELoss()

        # Metryki korelacji i dane epoki
        self.last_epoch = CheckpointInfo()
        self.best_epoch = CheckpointInfo()

        self.best_min_delta: float = 1e-4

        # Optymalizatora
        self.optimizer: Optimizer = Adam(
            params=self.model.parameters(),
            lr=training_config['training']['learning_rate']
        )

        # Logowanie
        self.log_writer = (
            SummaryWriter(str(self.logs_tensorboard_path))
            if training_config['training']['logging']['tensorboard']
            else None
        )

        # Data Loadery
        self.train_loader = build_data_loader(
            dataset_config=dataset_config,
            model_config=model_config,
            training_config=training_config,
            split_name='train',
            experiment_path=experiment_path
        )
        self.validation_loader = build_data_loader(
            dataset_config=dataset_config,
            model_config=model_config,
            training_config=training_config,
            split_name='validation',
            experiment_path=experiment_path
        )

        print('\n[Trainer] Załadowano eksperyment!')


    def train_one_epoch(self) -> float:
        """
        Wykonuje jedną epokę szkolenia.
        """
        self.model.train()

        running_loss = 0.0
        batch_count = 0

        for reference_image, distorted_image, ground_truth_score in self.train_loader:
            reference_image = reference_image.to(self.device)
            distorted_image = distorted_image.to(self.device)
            ground_truth_score = ground_truth_score.to(self.device)

            self.optimizer.zero_grad()

            prediction_score: Tensor = self.model(reference_image, distorted_image)
            loss: Tensor =  self.loss_function(prediction_score, ground_truth_score)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            batch_count += 1

        return running_loss / max(batch_count, 1)

    @torch.no_grad()
    def validate_last_epoch(self) -> tuple[float, CorrelationMetrics]:
        """
        Wykonuje jedną epokę walidacji.
        """

        self.model.eval()

        running_loss = 0.0
        batch_count = 0

        all_ground_truth_scores: list[float] = []
        all_predicted_scores: list[float] = []

        for reference_image, distorted_image, ground_truth_score in self.validation_loader:
            reference_image = reference_image.to(self.device)
            distorted_image = distorted_image.to(self.device)
            ground_truth_score = ground_truth_score.to(self.device)

            prediction_score: Tensor = self.model(reference_image, distorted_image)
            loss: Tensor = self.loss_function(prediction_score, ground_truth_score)

            running_loss += loss.item()
            batch_count += 1

            predicted_batch_values: list[float] = (
                prediction_score
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

            all_predicted_scores.extend(predicted_batch_values)
            all_ground_truth_scores.extend(ground_truth_batch_values)

        average_loss = running_loss / max(batch_count, 1)

        correlation_metrics = compute_correlations(
            ground_truth_scores=all_ground_truth_scores,
            predicted_scores=all_predicted_scores,
        )

        return average_loss, correlation_metrics


    def check_checkpoint_save_due(self) -> dict[str, bool]:
        save_checkpoint_triggers: dict[str, bool] = {
            'save_every_n_epochs': False,
            'save_last_epoch': False,
            'save_best_epoch': False
        }

        if (
                self.training_config['training']['checkpointing']['save_every_n_epochs'] > 0 and
                self.last_epoch.epoch % self.training_config['training']['checkpointing']['save_every_n_epochs'] == 0
        ):
            save_checkpoint_triggers['save_every_n_epochs'] = True

        if self.training_config['training']['checkpointing']['save_last_epoch']:
            save_checkpoint_triggers['save_last_epoch'] = True

        if self.training_config['training']['checkpointing']['save_best_epoch']:
            last_srcc = self.last_epoch.validation_correlation.srcc
            best_srcc = self.best_epoch.validation_correlation.srcc
            min_delta = self.best_min_delta

            if  last_srcc > best_srcc + min_delta:
                self.best_epoch = deepcopy(self.last_epoch)
                save_checkpoint_triggers['save_best_epoch'] = True

        return save_checkpoint_triggers


    def save_checkpoint(
            self,
            save_triggers: dict[str, bool],
    ) -> None:
        """
        Zapisuje wagi modelu do pliku.
        """

        if all(value is False for value in save_triggers.values()):
            print('WARNING: [Trainer] Nie przekazano żadnego powodu wykonania checkpointu!\n'
                  '    SPRAWDŹ POWÓD TEJ SYTUACJI! Checkpointy mogą być potencjalnie źle zapisywane!'
                  '    Aktualnie ustawiono domyślny powód: `save_last_epoch`.')

            save_triggers['save_last_epoch'] = True

        checkpoint_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'last_epoch': self.last_epoch,
            'best_epoch': self.best_epoch,
            'app_version': self.training_config['app_version'],
            'dataset_config': self.dataset_config,
            'model_config': self.model_config,
            'training_config': self.training_config
        }

        print(f"[Trainer] Zapisano checkpoint do:")

        if save_triggers['save_every_n_epochs']:
            epoch_checkpoint_path = self.checkpoints_path / f"epoch_{self.last_epoch.epoch}.pth"
            torch.save(checkpoint_dict, epoch_checkpoint_path)

            print(f"    {epoch_checkpoint_path}")

        if save_triggers['save_last_epoch']:
            last_checkpoint_path = self.checkpoints_path / 'last.pth'
            torch.save(checkpoint_dict, last_checkpoint_path)

            print(f"    {last_checkpoint_path}")

        if save_triggers['save_best_epoch']:
            best_checkpoint_path = self.checkpoints_path / 'best.pth'
            torch.save(checkpoint_dict, best_checkpoint_path)

            print(f"    {best_checkpoint_path}")


    def get_best_checkpoint(self) -> CheckpointPickle | None:
        best_checkpoint_path = self.checkpoints_path / 'best.pth'

        if not best_checkpoint_path.exists() or not best_checkpoint_path.is_file():
            return None

        return load_checkpoint_pickle(
            checkpoint_path=best_checkpoint_path,
            device=self.device,
            check_consistency=True
        )


    def load_checkpoint(self, checkpoint_path: Path) -> int:
        checkpoint_pickle = load_checkpoint_pickle(
            checkpoint_path=checkpoint_path,
            device=self.device,
            check_consistency=True
        )

        best_checkpoint_pickle = checkpoint_pickle

        checkpoint_name = checkpoint_path.name
        if checkpoint_name != 'best.pth':
            another_checkpoint_pickle = self.get_best_checkpoint()

            best_checkpoint_pickle = another_checkpoint_pickle or checkpoint_pickle

        self.best_epoch = best_checkpoint_pickle.best_epoch

        self.model.load_state_dict(checkpoint_pickle.model_state_dict)
        self.optimizer.load_state_dict(checkpoint_pickle.optimizer_state_dict)

        start_epoch = checkpoint_pickle.last_epoch.epoch + 1

        print(f"\n[Trainer] Wczytano checkpoint z {checkpoint_pickle.last_epoch.epoch}. epoki (kontynuacja od {start_epoch}. epoki).")

        return start_epoch


    def any_checkpoint_exists(self):
        if not self.checkpoints_path.exists() or not self.checkpoints_path.is_dir():
            raise FileNotFoundError(
                f"Error: Nie istnieje folder checkpointów\n"
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

        if last_checkpoint_exist and not self.training_config['training']['checkpointing']['save_last_epoch']:
            print(
                '\nWARNING: [Trainer] Znaleziono checkpoint `last.pth`.\n'
                '    Jednocześnie parametr konfiguracyjny `checkpointing.save_last_epoch` jest ustawiony na `false`.\n'
                '    Oznacza to, że przy aktualnej konfiguracji wskazany checkpoint nie mógł powstać.\n'
                '    Jeśli celowo umieściłeś plik `last.pth` w folderze `checkpoints/` i\n'
                '    jesteś pewien, że chcesz kontynuować trening, to wpisz [y / Y / t / T].\n'
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
                '    Jeśli stworzyłeś cały ten eksperyment na podstawie checkpointu z innego eksperymentu albo\n'
                '    celowo umieściłeś plik `init.pth` w folderze `checkpoints/` jako pretrenowany model i\n'
                '    jesteś pewien, że chcesz kontynuować trening, to wpisz [y / Y / t / T].\n'
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

    def train(self) -> None:
        num_of_epochs = self.training_config['training']['num_of_epochs']
        start_epoch = 1

        epoch_from_checkpoint = self.try_resume()

        if epoch_from_checkpoint > 1:
            start_epoch = epoch_from_checkpoint
        else:
            print(f"\n[Trainer] Startowanie treningu dla {num_of_epochs} epok...")

        for epoch in range(start_epoch, num_of_epochs+1):
            self.last_epoch.epoch = epoch

            print(f"\n[Trainer] Rozpoczęto trening {epoch}. epoki.")

            train_loss = self.train_one_epoch()
            validation_loss, validation_correlation = self.validate_last_epoch()

            self.last_epoch.train_loss = LossMetrics(mse=train_loss)
            self.last_epoch.validation_loss = LossMetrics(mse=validation_loss)
            self.last_epoch.validation_correlation = validation_correlation

            if self.log_writer:
                self.log_writer.add_scalar('loss/train', train_loss, epoch)
                self.log_writer.add_scalar('loss/validation', validation_loss, epoch)
                self.log_writer.add_scalar('correlation/plcc', validation_correlation.plcc, epoch)
                self.log_writer.add_scalar('correlation/srcc', validation_correlation.srcc, epoch)
                self.log_writer.add_scalar('correlation/krcc', validation_correlation.krcc, epoch)

            print(
                f"[Trainer] Ukończono trening epoki {epoch} / {num_of_epochs}    "
                f"|    Błąd trenowania: {train_loss:.4f}    "
                f"|    Błąd walidacji: {validation_loss:.4f}    "
                f"|    PLCC: {validation_correlation.plcc:.4f}    "
                f"|    SRCC: {validation_correlation.srcc:.4f}    "
                f"|    KRCC: {validation_correlation.krcc:.4f}"
            )

            save_checkpoint_triggers = self.check_checkpoint_save_due()

            if any(save_checkpoint_triggers.values()):
                self.save_checkpoint(save_triggers=save_checkpoint_triggers)

        if start_epoch <= num_of_epochs:
            print(f"\n[Trainer] Cały trening zakończony!\n"
                  f"    Przeprowadzono {num_of_epochs} epok szkolenia.\n")
        else:
            print(f"\n[Trainer] Nie można przeprowadzić treningu, ponieważ numer startowej epoki ({start_epoch})\n"
                  f"    jest większy niż maksymalna liczba epok treningu ({num_of_epochs}) w pliku konfiguracyjnym YAML eksperymentu.")

        input('\nNaciśnij Enter, aby zakończyć...')
