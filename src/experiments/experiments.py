from pathlib import Path
import shutil

from src.datasets.base_dataset import BaseDataset
from src.utils.configs import load_config
from src.datasets.factory import build_dataset
from src.datasets.splits import generate_split
from src.utils.paths import EXPERIMENTS_PATH


def _validate_experiment_creation_args(
        config_path: Path | None = None,
        checkpoint_path: Path | None = None
):
    if not EXPERIMENTS_PATH.parent.exists():
        raise FileNotFoundError('Error: Nie istnieje folder nadrzędny folderu `experiments`!')

    if config_path is None and checkpoint_path is None:
        raise ValueError('Error: Nie przekazano pliku konfiguracyjnego YAML ani pliku checkpointu z innego eksperymentu!')

    if config_path is not None and checkpoint_path is not None:
        raise ValueError(
            'Error: Przekazano jednocześnie plik konfiguracyjny YAML oraz plik checkpointu!\n'
            'Można jednocześnie przekazać tylko jeden z argumentów!'
        )

    if config_path is not None and not config_path.exists():
        raise FileNotFoundError(
            f"Error: Przekazany plik konfiguracyjny YAML nie istnieje!\n"
            f"Ścieżka: {config_path}"
        )

    if checkpoint_path is not None and not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Error: Przekazany plik checkpointu nie istnieje!\n"
            f"Ścieżka: {checkpoint_path}"
        )


def _create_directories(experiment_path: Path) -> None:
    directories = [
        experiment_path / 'checkpoints/',
        experiment_path / 'logs/',
        experiment_path / 'logs' / 'tensorboard/',
        experiment_path / 'splits/'
    ]

    for directory in directories:
        directory.mkdir(parents=False, exist_ok=False)


def _create_empty_files(experiment_path: Path) -> None:
    files = [
        experiment_path / 'logs' / 'train.log',
        experiment_path / 'metrics.csv',
        experiment_path / 'metrics.json',
        experiment_path / 'metrics.md',
        experiment_path / 'summary.md'
    ]

    for file in files:
        file.touch(exist_ok=False)


def create_experiment(
        experiment_name: str,
        config_path: Path | None = None,
        checkpoint_path: Path | None = None
) -> Path:
    _validate_experiment_creation_args(
        config_path=config_path,
        checkpoint_path=checkpoint_path
    )

    creating_from_checkpoint = checkpoint_path is not None

    EXPERIMENTS_PATH.mkdir(exist_ok=True)

    if creating_from_checkpoint:
        config_path = checkpoint_path.parent.parent / 'config.yaml'

        print(
            f"Tworzenie eksperymentu na podstawie istniejącego checkpointu:\n"
            f"    Ścieżka checkpointu: {checkpoint_path}\n"
            f"    Ścieżka configu związanego z checkpointem: {checkpoint_path}"
        )
    else:
        print(
            f"Tworzenie eksperymentu na podstawie pliku konfiguracyjnego YAML:\n"
            f"    Ścieżka: {config_path}"
        )

    config = load_config(config_path=config_path, check_consistency=True)

    dataset: BaseDataset = build_dataset(config=config)

    dataset_name = config['dataset']['name']

    experiment_path = EXPERIMENTS_PATH / dataset_name / experiment_name

    try:
        experiment_path.mkdir(parents=True, exist_ok=False)

        _create_directories(experiment_path=experiment_path)
        _create_empty_files(experiment_path=experiment_path)

        shutil.copy2(config_path, (experiment_path / 'config.yaml'))

        if creating_from_checkpoint:
            shutil.copytree(
                src=(checkpoint_path.parent.parent / 'splits/'),
                dst=(experiment_path / 'splits/'),
                dirs_exist_ok=True,
            )

            shutil.copy2(checkpoint_path, (experiment_path / 'checkpoints' / 'init.pth'))
        else:
            generate_split(
                dataset=dataset,
                train_split=config['training']['splits']['train'],
                validation_split=config['training']['splits']['validation'],
                test_split=config['training']['splits']['test'],
                random_seed=config['training']['splits']['random_seed'],
                output_directory=(experiment_path / 'splits/')
            )

        print(f"\nPoprawnie utworzono eksperyment `{experiment_name}`!")

        return experiment_path

    except FileExistsError:
        raise FileExistsError(
            f'Error: Eksperyment `{experiment_name}` dla datasetu `{dataset_name}` już istnieje!\n'
            f'Wybierz inną nazwę eksperymentu lub usuń istniejący folder.'
        )

    except PermissionError:
        raise PermissionError('Error: Brak uprawnień do utworzenia folderu eksperymentu!')

    except OSError as error:
        raise OSError(
            f'Error: Błąd systemu przy tworzeniu folderu eksperymentu!\n'
            f'Błąd: {error}'
        )

    finally:
        print(f'Ścieżka: {experiment_path}\n')
