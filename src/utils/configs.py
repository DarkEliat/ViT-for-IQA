from pathlib import Path
from typing import Any

import yaml

from src.utils.data_types import Config
from src.utils.paths import PROJECT_ROOT


def load_config(config_path: Path, check_consistency: bool) -> Config:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Error: Plik konfiguracyjny nie istnieje!\n"
            f"Ścieżka: {config_path}"
        )

    if config_path.suffix not in {'.yaml', '.yml'}:
        raise ValueError(
            f"Error: Nieprawidłowy format pliku konfiguracyjnego! Oczekiwano pliku YAML (.yaml / .yml)!\n"
            f"Ścieżka: {config_path}"

        )

    with open(config_path, 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise TypeError(
            f"Error: Nieprawidłowa struktura pliku YAML! Oczekiwano słownika (dict) na najwyższym poziomie!\n"
            f"Ścieżka: {config_path}"
        )

    if check_consistency:
        check_config_consistency(config=config, path=config_path)

    return config


def _get_section(config: Config, section_key: str) -> dict[str, Any]:
    if section_key not in config:
        raise ValueError(f"Error: Brakuje wymaganej sekcji `{section_key}` na najwyższym poziomie!")

    section_value = config[section_key]
    if not isinstance(section_value, dict):
        raise TypeError(f"Error: Sekcja `{section_key}` musi być słownikiem (dict)!")

    return section_value


def _require_keys(section_name: str, section: dict[str, Any], required_keys: set[str]) -> None:
    missing_keys = required_keys - section.keys()
    if missing_keys:
        raise ValueError(
            f"Error: Sekcja `{section_name}` jest niekompletna!\n"
            f"Brakuje: {sorted(missing_keys)}"
        )


def _check_dataset_section(dataset: dict[str, Any]) -> None:
    _require_keys(
        section_name='dataset',
        section=dataset,
        required_keys={
            'representative_name',
            'name',
            'length',
            'original_image_size',
            'reference_images_path',
            'distorted_images_path',
            'quality_label',
            'labels_path',
    })

    # --------------- representative_name ---------------
    if not isinstance(dataset['representative_name'], str) or dataset['representative_name'].strip() == '':
        raise ValueError('Error: `dataset.representative_name` musi być niepustym stringiem!')

    # --------------- name ---------------
    if not isinstance(dataset['name'], str) or dataset['name'].strip() == '':
        raise ValueError('Error: `dataset.name` musi być niepustym stringiem!')

    # --------------- length ---------------
    if not isinstance(dataset['length'], int) or dataset['length'] <= 0:
        raise ValueError('Error: `dataset.length` musi być dodatnią liczbą całkowitą!')

    # --------------- original_image_size ---------------
    original_image_size = dataset['original_image_size']
    if not isinstance(original_image_size, dict):
        raise TypeError('Error: `dataset.original_image_size` musi być słownikiem (dict)!')

    for dimension_name in ('width', 'height'):
        if (
            dimension_name not in original_image_size or
            not isinstance(original_image_size[dimension_name], int) or
            original_image_size[dimension_name] <= 0
        ):
            raise ValueError('Error: `dataset.original_image_size` musi zawierać dodatnie wartości `width` oraz `height`!')

    # --------------- reference_images_path ---------------
    reference_images_path = PROJECT_ROOT / dataset['reference_images_path']
    if not reference_images_path.exists() or not reference_images_path.is_dir():
        raise FileNotFoundError(
            f"Error: `dataset.reference_images_path` nie istnieje lub nie jest katalogiem!\n"
            f"Ścieżka: {reference_images_path}"
        )

    # --------------- distorted_images_path ---------------
    distorted_images_path = PROJECT_ROOT / dataset['distorted_images_path']
    if not distorted_images_path.exists() or not distorted_images_path.is_dir():
        raise FileNotFoundError(
            f"Error: `dataset.distorted_images_path` nie istnieje lub nie jest katalogiem!\n"
            f"Ścieżka: {distorted_images_path}"
        )

    # --------------- quality_label ---------------
    quality_label = dataset['quality_label']
    if not isinstance(quality_label, dict):
        raise TypeError('Error: `dataset.quality_label` musi być słownikiem (dict)!')

    _require_keys(
        section_name='dataset.quality_label',
        section=quality_label,
        required_keys={
            'type',
            'min',
            'max'
        }
    )

    if quality_label['type'] not in ('mos', 'dmos'):
        raise ValueError('Error: `dataset.quality_label.type` musi mieć wartość `mos` albo `dmos`!')

    if not isinstance(quality_label['min'], (int, float)) or not isinstance(quality_label['max'], (int, float)):
        raise TypeError('Error: `dataset.quality_label.min` oraz `dataset.quality_label.max` muszą być liczbami!')

    if quality_label['min'] >= quality_label['max']:
        raise ValueError('Error: `dataset.quality_label` ma niepoprawny zakres (`min` >= `max`)!')

    # --------------- labels_path ---------------
    labels_path = PROJECT_ROOT / dataset['labels_path']
    if not labels_path.exists() or not labels_path.is_file():
        raise FileNotFoundError(
            f"Error: `dataset.labels_path` nie istnieje lub nie jest plikiem!\n"
            f"Ścieżka: {labels_path}"
        )


def _check_model_section(model: dict[str, Any]) -> None:
    _require_keys(
        section_name='model',
        section=model,
        required_keys={
            'name',
            'input',
            'embedding_dimension',
            'output',
        }
    )

    # --------------- name ---------------
    if not isinstance(model['name'], str) or not model['name']:
        raise ValueError('Error: `model.name` musi być niepustym stringiem!')

    # --------------- input ---------------
    model_input = model['input']
    if not isinstance(model_input, dict):
        raise TypeError('Error: `model.input` musi być słownikiem (dict)!')

    _require_keys(
        section_name='model.input',
        section=model_input,
        required_keys={
            'type',
            'num_of_images',
            'have_same_image_size',
            'are_images_square',
            'image_size',
            'keep_original_aspect_ratio',
        },
    )

    if model_input['type'] != 'images':
        raise ValueError('Error: Aktualnie obsługiwany jest wyłącznie `model.input.type` == `images`!')

    if model_input['num_of_images'] != 2:
        raise ValueError('Error: FR-IQA wymaga dokładnie 2 obrazy (referencyjny + zniekształcony)!')

    for bool_key in ('have_same_image_size', 'are_images_square', 'keep_original_aspect_ratio'):
        if not isinstance(model_input[bool_key], bool):
            raise TypeError(f"Error: `model.input.{bool_key}` musi być typu bool!")

    image_size = model_input['image_size']

    if not isinstance(image_size, dict):
        raise TypeError('Error: `model.input.image_size` musi być słownikiem (dict)!')

    _require_keys(
        section_name='model.input.image_size',
        section=image_size,
        required_keys={
            'width',
            'height'
        }
    )

    for dimension_name in ('width', 'height'):
        if not isinstance(image_size[dimension_name], int) or image_size[dimension_name] <= 0:
            raise ValueError('Error: `model.input.image_size.width/height` muszą być dodatnimi liczbami całkowitymi!')

    if model_input['are_images_square'] and image_size['width'] != image_size['height']:
        raise ValueError('Error: `model.input.are_images_square == True`, ale `image_size.width` != `image_size.height`!')

    # --------------- embedding_dimension ---------------
    embedding_dimension = model['embedding_dimension']

    if not isinstance(embedding_dimension, int) or embedding_dimension <= 0 or embedding_dimension % 2 != 0:
        raise ValueError('Error: `model.embedding_dimension` musi być dodatnią parzystą liczbą całkowitą!')

    # --------------- output ---------------
    output_config = model['output']
    if not isinstance(output_config, dict):
        raise TypeError('Error: `model.output` musi być słownikiem (dict)!')

    _require_keys(
        section_name='model.output',
        section=output_config,
        required_keys={
            'type',
            'min',
            'max'
        }
    )

    if output_config['min'] != 0 or output_config['max'] != 1:
        raise ValueError('Error: Zakres `model.output` musi wynosić dokładnie [0, 1]!')

    if output_config['type'] not in ('normalized_mos', 'inverted_normalized_dmos'):
        raise ValueError('Error: `model.output.type` musi mieć wartość `normalized_mos` albo `inverted_normalized_dmos`!')


def _check_training_section(training: dict[str, Any]) -> None:
    _require_keys(
        section_name='training',
        section=training,
        required_keys={
            'quality_label',
            'split',
            'batch_size',
            'num_of_epochs',
            'early_stopping',
            'learning_rate',
            'device',
            'num_of_workers',
        },
    )

    # --------------- quality_label ---------------
    quality_label = training['quality_label']
    if not isinstance(quality_label, dict):
        raise TypeError('Error: `training.quality_label` musi być słownikiem (dict)!')

    _require_keys(
        section_name='training.quality_label',
        section=quality_label,
        required_keys={
            'type',
            'min',
            'max'
        }
    )

    # --------------- split ---------------
    split = training['split']
    if not isinstance(quality_label, dict):
        raise TypeError('Error: `training.split` musi być słownikiem (dict)!')

    _require_keys(
        section_name='training.quality_label',
        section=split,
        required_keys={
            'train_split',
            'random_seed',
        }
    )

    if not isinstance(training['split']['train_split'], (int, float)) or not 0 < training['split']['train_split'] < 1:
        raise ValueError('Error: `training.split.train_split` musi być z zakresu (0, 1)!')

    if not isinstance(training['split']['random_seed'], int) or not 0 < training['split']['random_seed'] >= 0:
        raise ValueError('Error: `training.split.train_split` musi być liczbą całkowitą większą lub równą 0!')

    # --------------- batch_size ---------------
    if not isinstance(training['batch_size'], int) or training['batch_size'] <= 0:
        raise ValueError('Error: `training.batch_size` musi być dodatnią liczbą całkowitą!')

    # --------------- num_of_epochs ---------------
    if not isinstance(training['num_of_epochs'], int) or training['num_of_epochs'] <= 0:
        raise ValueError('Error: `training.num_of_epochs` musi być dodatnią liczbą całkowitą!')

    # --------------- early_stopping ---------------
    early_stopping = training['early_stopping']

    _require_keys(
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


def _check_checkpointing_section(checkpointing: dict[str, Any]) -> None:
    _require_keys(
        section_name='checkpointing',
        section=checkpointing,
        required_keys={
            'enabled',
            'save_every_n_epochs',
            'save_last_epoch',
            'save_best_epoch',
        }
    )

    # --------------- enabled ---------------
    if not isinstance(checkpointing['enabled'], bool):
        raise TypeError('Error: `checkpointing.enabled` musi być typu bool!')

    # --------------- save_every_n_epochs ---------------
    save_every_n_epochs = checkpointing['save_every_n_epochs']

    if not isinstance(save_every_n_epochs, int):
        raise TypeError('Error: `checkpointing.save_every_n_epochs` musi być liczbą całkowitą!')

    if save_every_n_epochs <= 0:
        raise ValueError('Error: `checkpointing.save_every_n_epochs` musi być większe od 0!')

    # --------------- save_last_epoch ---------------
    # --------------- save_best_epoch ---------------
    for boolean_key in ('save_last_epoch', 'save_best_epoch'):
        if not isinstance(checkpointing[boolean_key], bool):
            raise TypeError(f"Error: `checkpointing.{boolean_key}` musi być typu bool!")


def _check_logging_section(logging: dict[str, Any]) -> None:
    _require_keys(
        section_name='logging',
        section=logging,
        required_keys={
            'tensorboard'
        }
    )

    # --------------- tensorboard ---------------
    if not isinstance(logging['tensorboard'], bool):
        raise TypeError('Error: `logging.tensorboard` musi być typu bool!')


def _check_cross_section_consistency(config: dict[str, Any]) -> None:
    dataset_quality_type = config['dataset']['quality_label']['type']
    model_output_type = config['model']['output']['type']
    training_quality_type = config['training']['quality_label']['type']

    expected_output_type_by_dataset_label_type = {
        'mos': 'normalized_mos',
        'dmos': 'inverted_normalized_dmos',
    }

    expected_type = expected_output_type_by_dataset_label_type[dataset_quality_type]

    if model_output_type != expected_type:
        raise ValueError(
            f"Error: Niespójność: `dataset.quality_label.type` nie pasuje do `model.output.type`!\n"
            f"  dataset.quality_label.type = {dataset_quality_type}\n"
            f"  model.output.type          = {model_output_type}\n"
            f"  expected                   = {expected_type}"
        )

    if training_quality_type != expected_type:
        raise ValueError(
            f"Error: Niespójność: `dataset.quality_label.type` nie pasuje do `training.quality_label.type`!\n"
            f"  dataset.quality_label.type    = {dataset_quality_type}\n"
            f"  training.quality_label.type   = {training_quality_type}\n"
            f"  expected                      = {expected_type}"
        )


def check_config_consistency(config: Config, path: Path | None = None) -> None:
    try:
        dataset_config = _get_section(config, section_key='dataset')
        model_config = _get_section(config, section_key='model')
        training_config = _get_section(config, section_key='training')
        checkpointing_config = _get_section(config, section_key='checkpointing')
        logging_config = _get_section(config, section_key='logging')

        _check_dataset_section(dataset_config)
        _check_model_section(model_config)
        _check_training_section(training_config)
        _check_checkpointing_section(checkpointing_config)
        _check_logging_section(logging_config)
        _check_cross_section_consistency(config)

    except Exception as error:
        raise RuntimeError(
            f"Error: Błąd weryfikacji spójności pliku konfiguracyjnego YAML!\n"
            f"Ścieżka: {path if path else 'NIEZNANA'}\n\n"
            f"{error}"
        ) from error
