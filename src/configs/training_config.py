from pathlib import Path
from typing import Any

from src.configs.validators import (
    require_config_keys,
    get_config_section,
    check_app_version_section
)
from src.datasets.dataset_map import DATASET_LIST
from src.models.model_map import MODEL_LIST
from src.utils.data_types import TrainingConfig


def _check_dataset_section(dataset: str) -> None:
    if dataset not in DATASET_LIST:
        raise ValueError(f"Error: Dataset o nazwie `{dataset}` nie jest dostępny!")


def _check_model_section(model: str) -> None:
    if model not in MODEL_LIST:
        raise ValueError(f"Error: Model o nazwie `{model}` nie jest dostępny!")


def _check_training_section(training: dict[str, Any]) -> None:
    require_config_keys(
        section_name='training',
        section=training,
        required_keys={
            'quality_label',
            'splits',
            'batch_size',
            'num_of_epochs',
            'early_stopping',
            'learning_rate',
            'device',
            'num_of_workers',
            'checkpointing',
            'logging'
        },
    )


    # --------------- quality_label ---------------
    quality_label = training['quality_label']
    if not isinstance(quality_label, dict):
        raise TypeError('Error: `training.quality_label` musi być słownikiem (dict)!')

    require_config_keys(
        section_name='training.quality_label',
        section=quality_label,
        required_keys={
            'type',
            'min',
            'max'
        }
    )


    # --------------- splits ---------------
    splits = training['splits']
    require_config_keys(
        section_name='training.splits',
        section=splits,
        required_keys={
            'train',
            'validation',
            'test',
            'random_seed',
        }
    )


    # --------------- splits.train ---------------
    # --------------- splits.validation ---------------
    # --------------- splits.test ---------------
    if not isinstance(training['splits']['train'], (int, float)) or not 0 < training['splits']['train'] < 1:
        raise ValueError('Error: `training.splits.train` musi być z zakresu (0, 1)!')

    if not isinstance(training['splits']['validation'], (int, float)) or not 0 < training['splits']['validation'] < 1:
        raise ValueError('Error: `training.splits.validation` musi być z zakresu (0, 1)!')

    if not isinstance(training['splits']['test'], (int, float)) or not 0 < training['splits']['test'] < 1:
        raise ValueError('Error: `training.splits.test` musi być z zakresu (0, 1)!')

    split_sum = training['splits']['train'] + training['splits']['validation'] + training['splits']['test']
    if abs(split_sum - 1.0) > 1e-9:
        raise ValueError(
            f"Error: Parametry `training.splits.train`, `training.splits.validation` oraz `training.splits.test` muszą sumować się do 1.0 (100%) !\n"
            f"Aktualna suma: {split_sum}"
        )


    # --------------- splits.random_seed ---------------
    if not isinstance(training['splits']['random_seed'], int) or not 0 < training['splits']['random_seed'] >= 0:
        raise ValueError('Error: `training.split.train_split` musi być liczbą całkowitą większą lub równą 0!')


    # --------------- batch_size ---------------
    if not isinstance(training['batch_size'], int) or training['batch_size'] <= 0:
        raise ValueError('Error: `training.batch_size` musi być dodatnią liczbą całkowitą!')


    # --------------- num_of_epochs ---------------
    if not isinstance(training['num_of_epochs'], int) or training['num_of_epochs'] <= 0:
        raise ValueError('Error: `training.num_of_epochs` musi być dodatnią liczbą całkowitą!')


    # --------------- early_stopping ---------------
    early_stopping = training['early_stopping']
    require_config_keys(
        section_name='training.early_stopping',
        section=early_stopping,
        required_keys={
            'enabled',
            'max_epochs_without_improvement',
            'min_improvement_delta',
        },
    )

    if not isinstance(early_stopping['enabled'], bool):
        raise TypeError('Error: `training.early_stopping.enabled` musi być typu bool!')

    if (
            not isinstance(early_stopping['max_epochs_without_improvement'], int) or
            early_stopping['max_epochs_without_improvement'] <= 0
    ):
        raise ValueError(
            'Error: `training.early_stopping.max_epochs_without_improvement` musi być dodatnią liczbą całkowitą!')

    if (
            not isinstance(early_stopping['min_improvement_delta'], (int, float)) or
            early_stopping['min_improvement_delta'] < 0
    ):
        raise ValueError('Error: `training.early_stopping.min_improvement_delta` musi być >= 0!')


    # --------------- learning_rate ---------------
    if not isinstance(training['learning_rate'], (int, float)) or training['learning_rate'] <= 0:
        raise ValueError('Error: `training.learning_rate` musi być > 0!')


    # --------------- device ---------------
    device = training['device']
    if not isinstance(device, str) or not device:
        raise ValueError('Error: `training.device` musi być niepustym stringiem!')

    if device not in ('cpu', 'cuda', 'cuda:0'):
        raise ValueError('Error: `training.device` musi mieć wartość: `cpu`, `cuda` lub `cuda:0`!')


    # --------------- num_of_workers ---------------
    if not isinstance(training['num_of_workers'], int) or training['num_of_workers'] < 0:
        raise ValueError('Error: `training.num_of_workers` musi być liczbą całkowitą >= 0!')


    # --------------- checkpointing ---------------
    checkpointing = training['checkpointing']
    require_config_keys(
        section_name='checkpointing',
        section=checkpointing,
        required_keys={
            'enabled',
            'save_every_n_epochs',
            'save_last_epoch',
            'save_best_epoch',
        }
    )

    # --------------- checkpointing.enabled ---------------
    if not isinstance(checkpointing['enabled'], bool):
        raise TypeError('Error: `checkpointing.enabled` musi być typu bool!')

    # --------------- checkpointing.save_every_n_epochs ---------------
    save_every_n_epochs = checkpointing['save_every_n_epochs']

    if not isinstance(save_every_n_epochs, int):
        raise TypeError('Error: `checkpointing.save_every_n_epochs` musi być liczbą całkowitą!')

    if save_every_n_epochs <= 0:
        raise ValueError('Error: `checkpointing.save_every_n_epochs` musi być większe od 0!')

    # --------------- checkpointing.save_last_epoch ---------------
    # --------------- checkpointing.save_best_epoch ---------------
    for boolean_key in ('save_last_epoch', 'save_best_epoch'):
        if not isinstance(checkpointing[boolean_key], bool):
            raise TypeError(f"Error: `checkpointing.{boolean_key}` musi być typu bool!")


    # --------------- logging ---------------
    logging = training['logging']
    require_config_keys(
        section_name='logging',
        section=logging,
        required_keys={
            'tensorboard'
        }
    )

    # --------------- logging.tensorboard ---------------
    if not isinstance(logging['tensorboard'], bool):
        raise TypeError('Error: `logging.tensorboard` musi być typu bool!')


def check_training_config_consistency(config: TrainingConfig, path: Path | None = None) -> None:
    try:
        app_version = get_config_section(config, section_key='app_version')
        dataset_config = get_config_section(config, section_key='dataset')
        model_config = get_config_section(config, section_key='model')
        training_config = get_config_section(config, section_key='training')

        check_app_version_section(app_version)
        _check_dataset_section(dataset_config)
        _check_model_section(model_config)
        _check_training_section(training_config)

    except Exception as error:
        raise RuntimeError(
            f"Error: Błąd weryfikacji spójności pliku konfiguracyjnego YAML!\n"
            f"Ścieżka: {path if path else 'NIEZNANA'}\n\n"
            f"{error}"
        ) from error
