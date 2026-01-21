import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from src.evaluation.correlation_metrics import CorrelationMetrics, compute_correlations
from src.inference.predictor import Predictor
from src.utils.data_types import SplitName, EvaluationResults
from src.datasets.factory import build_split_data_loader


class Evaluator:
    def __init__(
            self,
            experiment_path: Path,
            checkpoint_name: str = 'last.pth',
            split_name: SplitName = 'test',
            batch_size_override: int | None = None,
            num_of_workers_override: int | None = None
    ) -> None:
        self.experiment_path = experiment_path
        self.checkpoint_name = checkpoint_name
        self.split_name: SplitName = split_name

        self.checkpoint_path = experiment_path / 'checkpoints/'
        self.splits_path = experiment_path / 'splits/'
        self.metrics_json_path = experiment_path / 'metrics.json'
        self.metrics_csv_path = experiment_path / 'metrics.csv'
        self.summary_md_path = experiment_path / 'summary.md'

        self.predictor = Predictor(
            experiment_path=experiment_path,
            checkpoint_name=checkpoint_name,
            batch_size_override=batch_size_override,
            num_of_workers_override=num_of_workers_override
        )

        self.config = self.predictor.config

        self.config_name = self.config['config_name']
        self.dataset_name = self.config['dataset']['name']
        self.device = self.config['training']['device']

        self.batch_size: int = (
            batch_size_override
            if isinstance(batch_size_override, int)
            else self.config['training']['batch_size']
        )

        self.num_of_workers: int = (
            num_of_workers_override
            if isinstance(num_of_workers_override, int)
            else self.config['training']['num_of_workers']
        )

        self.data_loader = build_split_data_loader(
            config=self.config,
            split_name=split_name,
            experiment_path=experiment_path
        )

        print(
            "\n[Evaluator] Załadowano eksperyment do ewaluacji:\n"
            f"    Ścieżka eksperymentu: {self.experiment_path}\n"
            f"    Nazwa checkpoint: `{self.checkpoint_name}`\n"
            f"    Nazwa datasetu: `{self.dataset_name}`\n"
            f"    Nazwa splitu: `{self.split_name}`\n"
            f"    Nazwa configu: `{self.config_name}`\n"
            f"    Batch size: {self.batch_size}\n"
            f"    Device: `{self.device}`\n"
            f"    Number of workers: {self.num_of_workers}\n"
        )


    @torch.no_grad()
    def evaluate(
            self,
            apply_nonlinear_regression_for_plcc: bool = True,
            save_outputs: bool = True
    ):
        print('\n[Evaluator] Rozpoczęto ewaluację...')

        ground_truth_scores, predicted_scores = self.predictor.predict_with_ground_truth(
            data_loader=self.data_loader,
            enable_debug_batch_difference=False,
        )

        # Metryki korelacyjne IQA
        correlation_metrics: CorrelationMetrics = compute_correlations(
            ground_truth_scores=ground_truth_scores,
            predicted_scores=predicted_scores,
            apply_nonlinear_regression_for_plcc=apply_nonlinear_regression_for_plcc
        )

        # Metryki błędu (pomocnicze)
        ground_truth_array = np.asarray(ground_truth_scores, dtype=np.float64)
        predicted_array = np.asarray(predicted_scores, dtype=np.float64)

        if ground_truth_array.shape != predicted_array.shape:
            raise RuntimeError(
                f"Error: Niezgodność rozmiarów tablic!\n"
                f"    ground_truth_array.shape={ground_truth_array}\n"
                f"    predicted_array.shape={predicted_array.shape}\n"
                f"    To nie powinno się wydarzyć, jeśli data_loader i predykcje działają poprawnie."
            )

        errors_array = predicted_array - ground_truth_array

        mse_value = float(np.mean(np.square(errors_array)))
        rmse_value = float(np.mean(np.sqrt(mse_value)))
        mae_value = float(np.mean(np.abs(errors_array)))

        results = EvaluationResults(
            plcc=correlation_metrics.plcc,
            srcc=correlation_metrics.srcc,
            krcc=correlation_metrics.krcc,
            mse=mse_value,
            rmse=rmse_value,
            mae=mae_value,
            num_of_samples=len(ground_truth_scores),
            split_name=self.split_name,
            checkpoint_name=self.checkpoint_name,
            config_name=self.config_name,
            dataset_name=self.dataset_name,
            device=self.device
        )

        print(
            f"\n[Evaluator] Ukończono ewaluację:\n"
            f"    PLCC: {results.plcc:.6f}\n"
            f"    SRCC: {results.srcc:.6f}\n"
            f"    KRCC: {results.krcc:.6f}\n"
            f"    MSE: {results.mse:.6f}\n"
            f"    RMSE: {results.rmse:.6f}\n"
            f"    MAE: {results.mae:.6f}\n"
            f"    Liczba próbek: {results.num_of_samples}\n"
        )

        if save_outputs:
            self._save_results(results=results)

        return results


    def _save_results(self, results: EvaluationResults) -> None:
        self._save_results_to_json(results=results, output_path=self.metrics_json_path)
        self._save_results_to_csv(results=results, output_path=self.metrics_csv_path)
        self._save_results_to_markdown(results=results, output_path=self.summary_md_path)

        print(
            f"\n[Evaluator] Zapisano wyniki ewaluacji do:\n"
            f"    {self.metrics_json_path}\n"
            f"    {self.metrics_csv_path}\n"
            f"    {self.summary_md_path}\n"
        )


    @staticmethod
    def _save_results_to_json(results: EvaluationResults, output_path: Path) -> None:
        results_dictionary = asdict(results)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(results_dictionary, json_file, indent=4, ensure_ascii=False)


    @staticmethod
    def _save_results_to_csv(results: EvaluationResults, output_path: Path) -> None:
        results_dictionary = asdict(results)

        header_line = ','.join(results_dictionary.keys())
        values_line = ','.join(str(value) for value in results_dictionary.values())

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as csv_file:
            csv_file.write(header_line + '\n')
            csv_file.write(values_line + '\n')


    @staticmethod
    def _save_results_to_markdown(results: EvaluationResults, output_path: Path):
        markdown_lines = [
            "# Experiment summary",
            "",
            "## Identification:",
            f"- Dataset: `{results.dataset_name}`",
            f"- Config: `{results.config_name}`",
            f"- Split: `{results.split_name}`",
            f"- Checkpoint: `{results.checkpoint_name}`",
            f"- Device: `{results.device}`",
            f"- Number of samples: {results.num_of_samples}",
            "",
            "## Correlation metrics:",
            f"- PLCC: `{results.plcc:.6f}`",
            f"- SRCC: `{results.srcc:.6f}`",
            f"- KRCC: `{results.krcc:.6f}`",
            "",
            "## Error metrics:",
            f"- MSE: {results.mse:.6f}",
            f"- RMSE: {results.rmse:.6f}",
            f"- MAE: {results.mae:.6f}",
            ""
        ]

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as markdown_file:
            markdown_file.write('\n'.join(markdown_lines))






























