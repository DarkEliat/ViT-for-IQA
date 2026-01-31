import argparse
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

from src.cli.base_cli_command import BaseCliCommand
from src.cli.arguments import add_checkpoint_name_arg, add_experiment_path_arg, add_split_name_arg
from src.cli.validators import validate_experiment_path_arg, validate_split_name_arg
from src.datasets.factory import build_data_loader
from src.inference.predictor import Predictor
from src.utils.data_types import SplitName


@dataclass(frozen=True)
class PredictionCliArgs:
    experiment_path: Path
    checkpoint_name: str
    split_name: SplitName


class PredictionCliCommand(BaseCliCommand[PredictionCliArgs]):
    @property
    def command_name(self) -> str:
        return "run_prediction"


    @property
    def command_description(self) -> str:
        return "Uruchamia predykcję dla wskazanej nazwy checkpointu (.pth) wewnątrz ścieżki eksperymentu."


    def add_args(self, parser: argparse.ArgumentParser) -> None:
        add_experiment_path_arg(parser=parser, required=True)
        add_checkpoint_name_arg(parser=parser, required=True)
        add_split_name_arg(parser=parser, required=True)


    def validate_and_normalize_args(self, parsed_namespace: argparse.Namespace) -> PredictionCliArgs:
        experiment_path = validate_experiment_path_arg(experiment_path=parsed_namespace.experiment_path)
        checkpoint_name = parsed_namespace.checkpoint_name
        split_name = validate_split_name_arg(experiment_path=experiment_path, split_name=parsed_namespace.split_name)

        return PredictionCliArgs(
            experiment_path=experiment_path,
            checkpoint_name=checkpoint_name,
            split_name=split_name
        )


    def run_command(self, normalized_args: PredictionCliArgs) -> None:
        experiment_path = normalized_args.experiment_path
        checkpoint_name = normalized_args.checkpoint_name
        split_name = normalized_args.split_name

        predictor = Predictor(
            experiment_path=experiment_path,
            checkpoint_name=checkpoint_name,
        )

        data_loader = build_data_loader(
            config=predictor.config,
            split_name=split_name,
            experiment_path=experiment_path
        )

        predicted_scores = predictor.predict(data_loader=data_loader)

        print(f"\nPrzewidziane przez model wartości jakości dla kolejnych zniekształconych obrazów (split: {split_name}):")
        pprint(predicted_scores)
