from pathlib import Path
from typing import Any

from src.configs.validators import (
    require_config_keys,
    get_config_section,
    check_app_version_section
)
from src.utils.data_types import ModelConfig


def _check_model_section(model: dict[str, Any]) -> None:
    require_config_keys(
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
    require_config_keys(
        section_name='model.input',
        section=model_input,
        required_keys={
            'type',
            'num_of_images',
            'have_same_image_size',
            'are_images_square',
            'image_size',
            'keep_original_aspect_ratio',
        }
    )


    # --------------- input.type ---------------
    if model_input['type'] != 'images':
        raise ValueError('Error: Aktualnie obsługiwany jest wyłącznie `model.input.type` == `images`!')


    # --------------- input.num_of_images ---------------
    if model_input['num_of_images'] != 2:
        raise ValueError('Error: FR-IQA wymaga dokładnie 2 obrazy (referencyjny + zniekształcony)!')


    # --------------- input.image_size ---------------
    image_size = model_input['image_size']
    require_config_keys(
        section_name='model.input.image_size',
        section=image_size,
        required_keys={
            'width',
            'height'
        }
    )

    # --------------- input.image_size.width ---------------
    width = image_size['width']
    if not isinstance(width, int) or width <= 0:
        raise ValueError('Error: `model.input.image_size.width` musi być dodatnią liczbą całkowitą!')

    # --------------- input.image_size.height ---------------
    height = image_size['height']
    if not isinstance(height, int) or height <= 0:
        raise ValueError('Error: `model.input.image_size.height` musi być dodatnią liczbą całkowitą!')


    # --------------- input.have_same_image_size ---------------
    # --------------- input.are_images_square ---------------
    # --------------- input.keep_original_aspect_ratio ---------------
    for bool_key in ('have_same_image_size', 'are_images_square', 'keep_original_aspect_ratio'):
        if not isinstance(model_input[bool_key], bool):
            raise TypeError(f"Error: `model.input.{bool_key}` musi być typu bool!")

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

    require_config_keys(
        section_name='model.output',
        section=output_config,
        required_keys={
            'type',
            'min',
            'max'
        }
    )


    # --------------- output.min ---------------
    # --------------- output.max ---------------
    if output_config['min'] != 0 or output_config['max'] != 1:
        raise ValueError('Error: Zakres `model.output` musi wynosić dokładnie [0, 1]!')


    # --------------- output.type ---------------
    if output_config['type'] != 'unified_quality_score':
        raise ValueError('Error: `model.output.type` musi mieć wartość `normalized_mos` albo `inverted_normalized_dmos`!')


def check_model_config_consistency(config: ModelConfig, path: Path | None = None) -> None:
    try:
        app_version = get_config_section(config, section_key='app_version')
        model_config = get_config_section(config, section_key='model')

        check_app_version_section(app_version)
        _check_model_section(model_config)

    except Exception as error:
        raise RuntimeError(
            f"Error: Błąd weryfikacji spójności pliku konfiguracyjnego YAML!\n"
            f"Ścieżka: {path if path else 'NIEZNANA'}\n\n"
            f"{error}"
        ) from error
