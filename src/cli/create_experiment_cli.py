import argparse
from dataclasses import dataclass
from pathlib import Path

from src.cli.validators import validate_training_config_name_arg, validate_checkpoint_path_arg
from src.experiments.experiments import create_experiment
from src.cli.base_cli_command import BaseCliCommand
from src.cli.arguments import (
    add_experiment_name_arg,
    add_training_config_name_arg,
    add_checkpoint_path_arg
)
from src.utils.paths import CONFIGS_PATH, EXPERIMENTS_PATH


@dataclass(frozen=True)
class ExperimentCreationCliArgs:
    # Both modes:
    experiment_name: str

    # Mode `from_config`:
    training_config_name: str | None

    # Mode `from_checkpoint`:
    checkpoint_path: Path | None


class ExperimentCreationCliCommand( BaseCliCommand[ExperimentCreationCliArgs] ):
    @property
    def command_name(self) -> str:
        return "create_experiment"


    @property
    def command_description(self) -> str:
        return (
            "Tworzy nowy eksperyment na podstawie globalnego pliku konfiguracyjnego YAML z folderu `configs/`"
            "albo klonuje istniejący eksperyment (wybrany checkpoint (.pth), splity (.csv) oraz plik `summary.md`)."
        )


    def add_args(self, parser: argparse.ArgumentParser) -> None:
        add_experiment_name_arg(parser=parser, required=True)

        add_training_config_name_arg(parser=parser, required=False)
        add_checkpoint_path_arg(parser=parser, required=False)


    def validate_and_normalize_args(self, parsed_namespace: argparse.Namespace) -> ExperimentCreationCliArgs:
        experiment_name = parsed_namespace.experiment_name
        training_config_name = parsed_namespace.training_config_name
        checkpoint_path = parsed_namespace.checkpoint_path

        if training_config_name is not None:
            validate_training_config_name_arg(training_config_name=training_config_name)
        if checkpoint_path is not None:
            checkpoint_path = EXPERIMENTS_PATH / checkpoint_path
            validate_checkpoint_path_arg(checkpoint_path=checkpoint_path)

        is_config_mode = training_config_name is not None
        is_checkpoint_mode = checkpoint_path is not None

        proper_use_message = '''
        Użyj:\n
            --experiment-name <new_experiment_name>\n
                oraz\n
            --training-dataset_config-name <training_*.yaml>\n
        albo:\n
            --experiment-name <new_experiment_name>\n
                oraz\n
            --checkpoint-path <dataset_name>/<experiment_name>/checkpoints/<file.pth>
        '''

        if is_config_mode and is_checkpoint_mode:
            raise ValueError(
                f"Error: Nieprawidłowa kombinacja argumentów.\n"
                f"{proper_use_message}"
            )

        if not is_config_mode and not is_checkpoint_mode:
            raise ValueError(
                f"Error: Brakujące wymagane argumenty.\n"
                f"{proper_use_message}"
            )

        if is_config_mode:
            return ExperimentCreationCliArgs(
                experiment_name=experiment_name,
                training_config_name=training_config_name,
                checkpoint_path=None
            )

        return ExperimentCreationCliArgs(
            experiment_name=experiment_name,
            training_config_name=None,
            checkpoint_path=checkpoint_path
        )


    def run_command(self, normalized_args: ExperimentCreationCliArgs) -> None:
        experiment_name = normalized_args.experiment_name

        training_config_name = normalized_args.training_config_name
        training_config_path = CONFIGS_PATH / training_config_name

        checkpoint_path = normalized_args.checkpoint_path

        create_experiment(
            experiment_name=experiment_name,
            training_config_path=training_config_path,
            checkpoint_path=checkpoint_path
        )
