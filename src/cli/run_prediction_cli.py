import argparse
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

from src.cli.base_cli_command import BaseCliCommand
from src.cli.arguments import (
    add_split_name_arg,
    add_checkpoint_path_arg,
    add_dataset_name_arg, add_skip_checkpoint_consistency_check_arg
)
from src.cli.validators import (
    validate_split_name_arg,
    validate_checkpoint_path_arg,
    validate_dataset_name_arg
)
from src.configs.loader import load_config
from src.datasets.dataset_map import DATASET_NAME_TO_CONFIG_PATH_MAP
from src.datasets.factory import build_data_loader
from src.inference.predictor import Predictor
from src.utils.data_types import SplitName, DatasetName
from src.utils.paths import EXPERIMENTS_PATH


@dataclass(frozen=True)
class PredictionCliArgs:
    checkpoint_path: Path
    dataset_name: DatasetName | None
    split_name: SplitName | None
    skip_checkpoint_consistency_check: bool | None


class PredictionCliCommand(BaseCliCommand[PredictionCliArgs]):
    @property
    def command_name(self) -> str:
        return "run_prediction"


    @property
    def command_description(self) -> str:
        return "Uruchamia predykcję dla wskazanego checkpointu (.pth)."


    def add_args(self, parser: argparse.ArgumentParser) -> None:
        add_checkpoint_path_arg(parser=parser, required=True)
        add_dataset_name_arg(parser=parser, required=False)
        add_split_name_arg(parser=parser, required=False)
        add_skip_checkpoint_consistency_check_arg(parser=parser, required=False)


    def validate_and_normalize_args(self, parsed_namespace: argparse.Namespace) -> PredictionCliArgs:
        checkpoint_path = EXPERIMENTS_PATH / parsed_namespace.checkpoint_path
        dataset_name = parsed_namespace.dataset_name
        split_name = parsed_namespace.split_name
        skip_checkpoint_consistency_check = parsed_namespace.skip_checkpoint_consistency_check

        validate_checkpoint_path_arg(checkpoint_path=checkpoint_path)
        if dataset_name is not None:
            validate_dataset_name_arg(dataset_name=dataset_name)
        if split_name is not None:
            validate_split_name_arg(split_name=split_name)

        is_full_training_dataset_mode = dataset_name is not None
        is_split_cross_dataset_mode = split_name is not None

        proper_use_message = '''
        Użyj:\n
            --checkpoint-path <dataset_name>/<experiment_name>/checkpoints/<file.pth>\n
                oraz\n
            --dataset-name <dataset_name>\n
        albo:\n
            --checkpoint-path <dataset_name>/<experiment_name>/checkpoints/<file.pth>\n
                oraz\n
            --split-name <split_name>
        '''

        if is_full_training_dataset_mode and is_split_cross_dataset_mode:
            raise ValueError(
                f"Error: Nieprawidłowa kombinacja argumentów!\n"
                f"{proper_use_message}"
            )

        if not is_full_training_dataset_mode and not is_split_cross_dataset_mode:
            raise ValueError(
                f"Error: Brakujące wymagane argumenty!\n"
                f"{proper_use_message}"
            )

        return PredictionCliArgs(
            checkpoint_path=checkpoint_path,
            dataset_name=dataset_name,
            split_name=split_name,
            skip_checkpoint_consistency_check=skip_checkpoint_consistency_check
        )


    def run_command(self, normalized_args: PredictionCliArgs) -> None:
        checkpoint_path = normalized_args.checkpoint_path
        dataset_name = normalized_args.dataset_name
        split_name: SplitName | None = normalized_args.split_name
        skip_checkpoint_consistency_check: bool | None = normalized_args.skip_checkpoint_consistency_check

        predictor = Predictor(
            checkpoint_path=checkpoint_path,
            check_checkpoint_consistency=skip_checkpoint_consistency_check
        )

        if split_name is not None:
            experiment_path = checkpoint_path.parent.parent
            data_loader = build_data_loader(
                dataset_config=predictor.checkpoint_dataset_config,
                model_config=predictor.checkpoint_model_config,
                training_config=predictor.checkpoint_training_config,
                split_name=split_name,
                experiment_path=experiment_path
            )
        else:
            split_name: SplitName = 'full'

            dataset_config_path = DATASET_NAME_TO_CONFIG_PATH_MAP[dataset_name]
            dataset_config = load_config(
                config_path=dataset_config_path,
                config_type='dataset',
                check_consistency=True
            )

            data_loader = build_data_loader(
                dataset_config=dataset_config,
                model_config=predictor.checkpoint_model_config,
                training_config=predictor.checkpoint_training_config,
                split_name=split_name,
                experiment_path=None
            )

        predicted_scores = predictor.predict(data_loader=data_loader)

        print(f"\nPrzewidziane przez model wartości jakości dla kolejnych zniekształconych obrazów (split: {split_name}):")
        pprint(predicted_scores)
