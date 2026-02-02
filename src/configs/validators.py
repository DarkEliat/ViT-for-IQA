from typing import Any

from src.utils.data_types import DatasetConfig, ModelConfig, TrainingConfig


def get_config_section(
        config: DatasetConfig | ModelConfig | TrainingConfig,
        section_key: str
) -> dict[str, Any] | float | int | str:
    if section_key not in config:
        raise ValueError(f"Error: Brakuje wymaganej sekcji `{section_key}` na najwyższym poziomie!")

    section_value = config[section_key]

    return section_value


def require_config_keys(section_name: str, section: dict[str, Any], required_keys: set[str]) -> None:
    if not isinstance(section, dict):
        raise TypeError(f"Sekcja `{section_name}` powinna być słownikiem (dict)!")

    missing_keys = required_keys - section.keys()
    if missing_keys:
        raise ValueError(
            f"Error: Sekcja `{section_name}` jest niekompletna!\n"
            f"Brakuje: {sorted(missing_keys)}"
        )


# TODO: Uzupełnić sprawdzanie numery wersji pliku konfiguracyjnego z wersją aplikacji zapisaną w `.app_version`
def check_app_version_section(app_version: str) -> None:
    ...


def check_cross_section_consistency(
        dataset_config: DatasetConfig,
        model_config: ModelConfig,
        training_config: TrainingConfig,
) -> None:
    dataset_quality_type = dataset_config['dataset']['quality_label']['type']
    model_output_type = model_config['model']['output']['type']
    training_quality_type = training_config['training']['quality_label']['type']

    expected_type = 'unified_quality_score'

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
