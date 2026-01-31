import argparse
from dataclasses import dataclass
from pathlib import Path

from src.cli.base_cli_command import BaseCliCommand
from src.cli.arguments import (
    add_checkpoint_name_arg,
    add_experiment_path_arg,
    add_split_name_arg,
)
from src.cli.validators import (
    validate_experiment_path_arg,
    validate_split_name_arg,
)
from src.evaluation.evaluator import Evaluator
from src.utils.data_types import SplitName


@dataclass(frozen=True)
class EvaluationCliArgs:
    experiment_path: Path
    checkpoint_name: str
    split_name: SplitName


class EvaluationCliCommand(BaseCliCommand[EvaluationCliArgs]):
    @property
    def command_name(self) -> str:
        return "run_evaluation"


    @property
    def command_description(self) -> str:
        return "Uruchamia ewaluację dla wskazanej checkpointu (.pth) i splitu wewnątrz ścieżki eksperymentu."


    def add_args(self, parser: argparse.ArgumentParser) -> None:
        add_experiment_path_arg(parser=parser, required=True)
        add_split_name_arg(parser=parser, required=True)
        add_checkpoint_name_arg(parser=parser, required=True)


    def validate_and_normalize_args(self, parsed_namespace: argparse.Namespace) -> EvaluationCliArgs:
        experiment_path = validate_experiment_path_arg(experiment_path=parsed_namespace.experiment_path)
        split_name = parsed_namespace.split_name
        checkpoint_name = parsed_namespace.checkpoint_name

        validate_split_name_arg(experiment_path=experiment_path, split_name=split_name)
        resolve_checkpoint_path(experiment_path=experiment_path, checkpoint_name=checkpoint_name)

        return EvaluationCliArgs(
            experiment_path=experiment_path,
            checkpoint_name=checkpoint_name,
            split_name=split_name
        )


    def run_command(self, normalized_args: EvaluationCliArgs) -> None:
        evaluator = Evaluator(
            experiment_path=normalized_args.experiment_path,
            checkpoint_name=normalized_args.checkpoint_name,
            split_name=normalized_args.split_name
        )

        evaluator.evaluate(save_outputs=False)
