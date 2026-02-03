import argparse
from dataclasses import dataclass
from pathlib import Path

from src.cli.base_cli_command import BaseCliCommand
from src.cli.arguments import add_experiment_path_arg, add_skip_checkpoint_consistency_check_arg
from src.cli.validators import validate_experiment_path_arg
from src.training.trainer import Trainer
from src.utils.paths import EXPERIMENTS_PATH


@dataclass(frozen=True)
class TrainingCliArgs:
    experiment_path: Path
    skip_checkpoint_consistency_check: bool | None


class TrainingCliCommand(BaseCliCommand[TrainingCliArgs]):
    @property
    def command_name(self) -> str:
        return "run_training"


    @property
    def command_description(self) -> str:
        return "Uruchamia trening we wskazanej ścieżce eksperymentu."


    def add_args(self, parser: argparse.ArgumentParser) -> None:
        add_experiment_path_arg(parser=parser, required=True)
        add_skip_checkpoint_consistency_check_arg(parser=parser, required=False)


    def validate_and_normalize_args(self, parsed_namespace: argparse.Namespace) -> TrainingCliArgs:
        experiment_path = EXPERIMENTS_PATH / parsed_namespace.experiment_path
        skip_checkpoint_consistency_check = parsed_namespace.skip_checkpoint_consistency_check

        validate_experiment_path_arg(experiment_path=experiment_path)

        return TrainingCliArgs(
            experiment_path=experiment_path,
            skip_checkpoint_consistency_check=skip_checkpoint_consistency_check
        )


    def run_command(self, normalized_args: TrainingCliArgs) -> None:
        experiment_path = normalized_args.experiment_path
        skip_checkpoint_consistency_check = normalized_args.skip_checkpoint_consistency_check

        trainer = Trainer(experiment_path=experiment_path)
        trainer.train(check_checkpoint_consistency=skip_checkpoint_consistency_check)
