import argparse
from dataclasses import dataclass
from pathlib import Path

from src.experiments.experiments import create_experiment
from src.cli.base_cli_command import BaseCliCommand
from src.cli.arguments import (
    add_checkpoint_name_arg,
    add_experiment_path_arg,
    add_global_config_name_arg,
    add_experiment_name_arg
)
from src.cli.validators import validate_experiment_path_arg
from src.utils.paths import CONFIGS_PATH


@dataclass(frozen=True)
class ExperimentCreationCliArgs:
    # Both modes:
    experiment_name: str

    # Mode `from_global_config`:
    global_config_name: str | None

    # Mode `from_checkpoint`:
    experiment_path: Path | None
    checkpoint_name: str | None


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

        add_global_config_name_arg(parser=parser, required=False)

        add_experiment_path_arg(parser=parser, required=False)
        add_checkpoint_name_arg(parser=parser, required=False)


    def validate_and_normalize_args(self, parsed_namespace: argparse.Namespace) -> ExperimentCreationCliArgs:
        experiment_name = parsed_namespace.experiment_name

        global_config_name = parsed_namespace.global_config_name

        experiment_path = parsed_namespace.experiment_path
        checkpoint_name = parsed_namespace.checkpoint_name

        is_global_config_mode = global_config_name is not None
        is_checkpoint_mode = (experiment_path is not None) and (checkpoint_name is not None)

        if is_global_config_mode and is_checkpoint_mode:
            raise ValueError(
                'Error: Nieprawidłowa kombinacja argumentów.\n'
                'Użyj:\n'
                '    --global-config <config.yaml>\n'
                'albo:\n'
                '    --experiment-path <directory path> --checkpoint <file.pth>\n'
                'ale nie obie kombinacje na raz.'
            )

        if not is_global_config_mode and not is_checkpoint_mode:
            raise ValueError(
                'Error: Brakujące wymagane argumenty.\n'
                'Użyj:\n'
                '    --global-config <config.yaml>\n'
                'albo:\n'
                '    --experiment-path <directory path> --checkpoint <file.pth>'
            )

        if is_global_config_mode:
            return ExperimentCreationCliArgs(
                experiment_name=experiment_name,
                global_config_name=global_config_name,
                experiment_path=None,
                checkpoint_name=None
            )

        experiment_path = validate_experiment_path_arg(experiment_path=experiment_path)

        return ExperimentCreationCliArgs(
            experiment_name=experiment_name,
            global_config_name=None,
            experiment_path=experiment_path,
            checkpoint_name=checkpoint_name
        )


    def run_command(self, normalized_args: ExperimentCreationCliArgs) -> None:
        experiment_name = normalized_args.experiment_name
        global_config_name = normalized_args.global_config_name
        experiment_path = normalized_args.experiment_path
        checkpoint_name = normalized_args.checkpoint_name

        global_config_path = (CONFIGS_PATH / global_config_name) if global_config_name else None
        checkpoint_path = (experiment_path / 'checkpoints' / checkpoint_name) if experiment_path and checkpoint_name else None

        create_experiment(
            experiment_name=experiment_name,
            config_path=global_config_path,
            checkpoint_path=checkpoint_path
        )
