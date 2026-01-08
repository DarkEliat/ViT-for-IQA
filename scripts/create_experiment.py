import os
from pathlib import Path
import shutil
from typing import Any

import yaml

from src.utils.configs import check_config_consistency, load_config
from src.utils.data_types import Config
from src.utils.paths import CONFIGS_PATH, EXPERIMENTS_PATH
from src.utils.dataset_splits import generate_and_save_dataset_split


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def get_global_config_paths() -> list[Path]:
    if not CONFIGS_PATH.exists() or not CONFIGS_PATH.is_dir():
        raise FileNotFoundError('Error: Folder `configs` z plikami konfiguracyjnymi nie istnieje!')

    config_files = list(CONFIGS_PATH.glob('*.yaml'))

    return sorted(config_files)


def menu(config_files: list[Path]) -> Path:
    if not config_files:
        raise FileNotFoundError('Error: Nie znaleziono żadnego pliku konfiguracyjnego! Dodaj plik .yaml do `config`!')

    print('Tworzenie eksperymentu...\n'
          'Wybierz dostępny plik konfiguracyjny na podstawie, którego szkolony będzie model:\n')

    for i, config_file in enumerate(config_files):
        print(f'{i + 1}. {config_file.name}')

    while True:
        try:
            choice = input(f'\nPodaj numer pliku (1-{len(config_files)}): ')
            chosen_option = int(choice) - 1

            if 0 <= chosen_option < len(config_files):
                return config_files[chosen_option]
            else:
                print(f'Error: Nieprawidłowy numer! Wybierz 1-{len(config_files)}')
        except ValueError:
            print('Error: Podaj poprawną liczbę!')


def get_experiment_name_from_user() -> str:
    return input('\nPodaj nazwę nowego eksperymentu: ')


def create_experiment(
        experiment_name: str,
        config_path: Path | None = None,
        checkpoint_path: Path | None = None
) -> Path:
    if not EXPERIMENTS_PATH.parent.exists():
        raise FileNotFoundError('Error: Nie istnieje folder nadrzędny folderu `experiments`!')

    if (
        (config_path is None or
         not config_path.exists())
            and
        (checkpoint_path is None or
         not checkpoint_path.exists())
    ):
        raise FileNotFoundError('Error: Nie przekazano pliku konfiguracyjnego YAML ani pliku checkpointu z innego eksperymentu\n'
                                'albo przekazany plik nie istnieje!')

    if (config_path is not None and
        checkpoint_path is not None):
        raise ValueError('Error: Przekazano jednocześnie plik konfiguracyjny YAML oraz plik checkpointu!\n'
                         'Można jednocześnie przekazać tylko jeden z argumentów!')

    EXPERIMENTS_PATH.mkdir(exist_ok=True)

    if config_path is None or not config_path.exists():
        config_path = checkpoint_path.parent.parent / 'config.yaml'

    with open(config_path, 'r') as config_file:
        config: Config = yaml.safe_load(config_file)

    dataset_name = config['dataset']['name']

    experiment_path = EXPERIMENTS_PATH / f"{dataset_name}" / f"{experiment_name}/"

    folders_to_create = [
        experiment_path / 'checkpoints/',
        experiment_path / 'logs/',
        experiment_path / 'logs' / 'tensorboard/',
        experiment_path / 'splits/'
    ]

    empty_files_to_create = [
        experiment_path / 'logs' / 'train.log',
        experiment_path / 'metrics.csv',
        experiment_path / 'metrics.json',
        experiment_path / 'summary.md'
    ]

    try:
        experiment_path.mkdir(parents=True, exist_ok=False)

        for folder in folders_to_create:
            folder.mkdir(parents=True, exist_ok=True)

        for file in empty_files_to_create:
            file.touch(exist_ok=False)

        shutil.copy2(config_path, (experiment_path / 'config.yaml'))

        if checkpoint_path is None:
            generate_and_save_dataset_split(
                dataset_length=config['dataset']['length'],
                train_split=config['training']['split']['train_split'],
                output_directory=(experiment_path / 'splits/'),
                random_seed=config['training']['split']['random_seed']
            )
        else:
            shutil.copytree(
                src=checkpoint_path.parent.parent / 'splits/',
                dst=experiment_path / 'splits/',
                dirs_exist_ok=True,
            )

            shutil.copy2(checkpoint_path, (experiment_path / 'checkpoints' / 'init.pth'))
            shutil.copy2(checkpoint_path, (experiment_path / 'summary.md'))

            print(f"\nPoprawnie utworzono eksperyment `{experiment_name}`!")

        return experiment_path
    except FileExistsError:
        raise FileExistsError(
            f'Error: Eksperyment `{experiment_name}` dla datasetu `{dataset_name}` już istnieje!\n'
            f'Wybierz inną nazwę eksperymentu lub usuń istniejący folder.'
        )
    except PermissionError:
        raise PermissionError(
            f'Error: Brak uprawnień do utworzenia folderu eksperymentu!'
        )
    except OSError as error:
        raise OSError(
            f'Error: Błąd systemu przy tworzeniu folderu eksperymentu!\n'
            f'Błąd: {error}'
        )
    finally:
        print(f'Ścieżka: {experiment_path}\n')


if __name__ == '__main__':
    all_global_config_paths = get_global_config_paths()

    chosen_config_path = menu(config_files=all_global_config_paths)

    experiment_name = get_experiment_name_from_user()

    create_experiment(config_path=chosen_config_path, experiment_name=experiment_name)
