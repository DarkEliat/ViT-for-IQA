from pathlib import Path
from typing import Any

from src.configs.validators import (
    require_config_keys,
    get_config_section,
    check_app_version_section
)
from src.utils.data_types import DatasetConfig
from src.utils.paths import PROJECT_ROOT_PATH


def _check_dataset_section(dataset: dict[str, Any]) -> None:
    require_config_keys(
        section_name='dataset',
        section=dataset,
        required_keys={
            'representative_name',
            'name',
            'original_image_size',
            'images',
            'quality_label',
            'labels_path',
        }
    )


    # --------------- representative_name ---------------
    if not isinstance(dataset['representative_name'], str) or dataset['representative_name'].strip() == '':
        raise ValueError('Error: `dataset.representative_name` musi być niepustym stringiem!')


    # --------------- name ---------------
    if not isinstance(dataset['name'], str) or dataset['name'].strip() == '':
        raise ValueError('Error: `dataset.name` musi być niepustym stringiem!')


    # --------------- original_image_size ---------------
    original_image_size = dataset['original_image_size']
    require_config_keys(
        section_name='dataset.original_image_size',
        section=original_image_size,
        required_keys={
            'width',
            'height'
        }
    )


    # --------------- original_image_size.width ---------------
    width = original_image_size['width']
    if not isinstance(width, int) or width <= 0:
        raise ValueError('Error: `dataset.original_image_size.width` musi być dodatnią liczbą całkowitą!')


    # --------------- original_image_size.height ---------------
    height = original_image_size['height']
    if not isinstance(height, int) or height <= 0:
        raise ValueError('Error: `dataset.original_image_size.height` musi być dodatnią liczbą całkowitą!')


    # --------------- images ---------------
    images = dataset['images']
    require_config_keys(
        section_name='dataset.images',
        section=images,
        required_keys={
            'reference',
            'distorted'
        }
    )


    # --------------- images.reference ---------------
    reference = images['reference']
    require_config_keys(
        section_name='dataset.images.reference',
        section=reference,
        required_keys={
            'path',
            'count'
        }
    )


    # --------------- images.reference.path ---------------
    reference_images_path = PROJECT_ROOT_PATH / reference['path']
    if not reference_images_path.exists() or not reference_images_path.is_dir():
        raise FileNotFoundError(
            f"Error: `dataset.images.reference.path` nie istnieje lub nie jest katalogiem!\n"
            f"Ścieżka: {reference_images_path}"
        )


    # --------------- images.reference.count ---------------
    reference_images_count = reference['count']
    if reference_images_count < 1:
        raise FileNotFoundError('Error: `dataset.images.reference.count` musi być dodatnią liczbą całkowitą!')


    # --------------- images.distorted ---------------
    distorted = images['distorted']
    require_config_keys(
        section_name='dataset.images.distorted',
        section=distorted,
        required_keys={
            'path',
            'count'
        }
    )

    # --------------- images.distorted.path ---------------
    distorted_images_path = PROJECT_ROOT_PATH / distorted['path']
    if not distorted_images_path.exists() or not distorted_images_path.is_dir():
        raise FileNotFoundError(
            f"Error: `dataset.images.distorted.path` nie istnieje lub nie jest katalogiem!\n"
            f"Ścieżka: {distorted_images_path}"
        )


    # --------------- images.distorted.count ---------------
    distorted_images_count = distorted['count']
    if distorted_images_count < 1:
        raise FileNotFoundError('Error: `dataset.images.distorted.count` musi być dodatnią liczbą całkowitą!')


    # --------------- quality_label ---------------
    quality_label = dataset['quality_label']
    if not isinstance(quality_label, dict):
        raise TypeError('Error: `dataset.quality_label` musi być słownikiem (dict)!')

    require_config_keys(
        section_name='dataset.quality_label',
        section=quality_label,
        required_keys={
            'type',
            'min',
            'max'
        }
    )


    # --------------- quality_label.type ---------------
    if quality_label['type'] not in ('mos', 'dmos'):
        raise ValueError('Error: `dataset.quality_label.type` musi mieć wartość `mos` albo `dmos`!')


    # --------------- quality_label.min ---------------
    # --------------- quality_label.max ---------------
    if not isinstance(quality_label['min'], (int, float)) or not isinstance(quality_label['max'], (int, float)):
        raise TypeError('Error: `dataset.quality_label.min` oraz `dataset.quality_label.max` muszą być liczbami!')

    if quality_label['min'] >= quality_label['max']:
        raise ValueError('Error: `dataset.quality_label` ma niepoprawny zakres (`min` >= `max`)!')


    # --------------- labels_path ---------------
    labels_path = PROJECT_ROOT_PATH / dataset['labels_path']
    if not labels_path.exists() or not labels_path.is_file():
        raise FileNotFoundError(
            f"Error: `dataset.labels_path` nie istnieje lub nie jest plikiem!\n"
            f"Ścieżka: {labels_path}"
        )


def check_dataset_config_consistency(config: DatasetConfig, path: Path | None = None) -> None:
    try:
        app_version = get_config_section(config, section_key='app_version')
        dataset_config = get_config_section(config, section_key='dataset')

        check_app_version_section(app_version)
        _check_dataset_section(dataset_config)

    except Exception as error:
        raise RuntimeError(
            f"Error: Błąd weryfikacji spójności pliku konfiguracyjnego YAML!\n"
            f"Ścieżka: {path if path else 'NIEZNANA'}\n\n"
            f"{error}"
        ) from error
