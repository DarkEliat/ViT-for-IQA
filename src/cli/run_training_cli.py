import argparse
from dataclasses import dataclass
from pathlib import Path

from src.cli.base_cli_command import BaseCliCommand
from src.cli.arguments import add_experiment_path_arg
from src.cli.validators import validate_experiment_path_arg
from src.training.trainer import Trainer


@dataclass(frozen=True)
class TrainingCliArgs:
    experiment_path: Path


class TrainingCliCommand(BaseCliCommand[TrainingCliArgs]):
    @property
    def command_name(self) -> str:
        return "run_training"


    @property
    def command_description(self) -> str:
        return "Uruchamia trening we wskazanej ścieżce eksperymentu."


    def add_args(self, parser: argparse.ArgumentParser) -> None:
        add_experiment_path_arg(parser=parser, required=True)


    def validate_and_normalize_args(self, parsed_namespace: argparse.Namespace) -> TrainingCliArgs:
        experiment_path = validate_experiment_path_arg(experiment_path=parsed_namespace.experiment_path)
        return TrainingCliArgs(experiment_path=experiment_path)


    def run_command(self, normalized_args: TrainingCliArgs) -> None:
        trainer = Trainer(experiment_path=normalized_args.experiment_path)
        trainer.train()
