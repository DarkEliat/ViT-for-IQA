import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluation.correlation_metrics import CorrelationMetrics, compute_correlations
from src.inference.predictor import Predictor
from src.utils.data_types import SplitName, EvaluationResults, LossMetrics


class Evaluator:
    def __init__(
            self,
            checkpoint_path: Path,
            check_checkpoint_consistency: bool = True,
            batch_size_override: int | None = None,
            num_of_workers_override: int | None = None
    ) -> None:
        print(f"\n[Evaluator] Rozpoczęto ładowanie checkpointu...")

        experiment_path = checkpoint_path.parent.parent
        checkpoint_name = checkpoint_path.name

        self.experiment_path = experiment_path
        self.checkpoint_name = checkpoint_name

        self.checkpoint_path = experiment_path / 'checkpoints/'
        self.splits_path = experiment_path / 'splits/'
        self.metrics_json_path = experiment_path / 'metrics.json'
        self.metrics_csv_path = experiment_path / 'metrics.csv'
        self.metrics_md_path = experiment_path / 'metrics.md'
        self.summary_md_path = experiment_path / 'summary.md'

        self.predictor = Predictor(
            checkpoint_path=checkpoint_path,
            check_checkpoint_consistency=check_checkpoint_consistency,
            batch_size_override=batch_size_override,
            num_of_workers_override=num_of_workers_override
        )

        self.checkpoint_dataset_config = self.predictor.checkpoint_dataset_config
        self.checkpoint_model_config = self.predictor.checkpoint_model_config
        self.checkpoint_training_config = self.predictor.checkpoint_training_config

        self.checkpoint_training_config_name = self.checkpoint_training_config['config_name']
        self.checkpoint_dataset_name = self.checkpoint_dataset_config['dataset']['name']
        self.device = self.checkpoint_training_config['training']['device']

        self.batch_size: int = (
            batch_size_override
            if isinstance(batch_size_override, int)
            else self.checkpoint_training_config['training']['batch_size']
        )

        self.num_of_workers: int = (
            num_of_workers_override
            if isinstance(num_of_workers_override, int)
            else self.checkpoint_training_config['training']['num_of_workers']
        )

        print(
            "\n[Evaluator] Załadowano checkpoint:\n"
            f"    Ścieżka eksperymentu: {self.experiment_path}\n"
            f"    Nazwa checkpointu, który zostanie poddany ewaluacji: `{self.checkpoint_name}`\n"
            f"    Dane o dokonanym treningu wskazanego checkpointu:\n"
            f"        Nazwa datasetu treningowego: `{self.checkpoint_dataset_name}`\n"
            f"        Nazwa configu treningowego: `{self.checkpoint_training_config_name}`\n"
            f"        Batch size: {self.batch_size}\n"
            f"        Device: `{self.device}`\n"
            f"        Number of workers: {self.num_of_workers}"
        )


    @torch.no_grad()
    def evaluate(
            self,
            data_loader: DataLoader,
            split_name: SplitName,
            save_outputs: bool = False
    ):
        print('\n[Evaluator] Rozpoczęto ewaluację...')

        ground_truth_scores, predicted_scores = self.predictor.predict_with_ground_truth(
            data_loader=data_loader,
        )

        # Metryki korelacyjne IQA
        correlation_metrics: CorrelationMetrics = compute_correlations(
            ground_truth_scores=ground_truth_scores,
            predicted_scores=predicted_scores,
        )

        # Metryki błędu (pomocnicze)
        ground_truth_array = np.asarray(ground_truth_scores, dtype=np.float64)
        predicted_array = np.asarray(predicted_scores, dtype=np.float64)

        if ground_truth_array.shape != predicted_array.shape:
            raise RuntimeError(
                f"Error: Niezgodność rozmiarów tablic!\n"
                f"    ground_truth_array.shape={ground_truth_array.shape}\n"
                f"    predicted_array.shape={predicted_array.shape}\n"
                f"    To nie powinno się wydarzyć, jeśli data_loader i predykcje działają poprawnie."
            )

        errors_array = predicted_array - ground_truth_array

        mse_value = float(np.mean(np.square(errors_array)))
        rmse_value = float(np.sqrt(mse_value))
        mae_value = float(np.mean(np.abs(errors_array)))

        results = EvaluationResults(
            correlation=correlation_metrics,
            loss=LossMetrics(
                mse=mse_value,
                rmse=rmse_value,
                mae=mae_value
            ),
            num_of_samples=len(ground_truth_scores),
            split_name=split_name,
            checkpoint_name=self.checkpoint_name,
            training_config_name=self.checkpoint_training_config_name,
            dataset_name=self.checkpoint_dataset_name,
            device=self.device
        )

        print(
            f"\n[Evaluator] Ukończono ewaluację:\n"
            f"    PLCC: {results.correlation.plcc:.6f}\n"
            f"    SRCC: {results.correlation.srcc:.6f}\n"
            f"    KRCC: {results.correlation.krcc:.6f}\n"
            f"    MSE: {results.loss.mse:.6f}\n"
            f"    RMSE: {results.loss.rmse:.6f}\n"
            f"    MAE: {results.loss.mae:.6f}\n"
            f"    Liczba próbek: {results.num_of_samples}\n"
        )

        if save_outputs:
            self._save_results(results=results)

        return results


    def _save_results(self, results: EvaluationResults) -> None:
        self._save_results_to_json(results=results, output_path=self.metrics_json_path)
        self._save_results_to_csv(results=results, output_path=self.metrics_csv_path)
        self._save_results_to_markdown(results=results, output_path=self.metrics_md_path)

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
            f"- Config: `{results.training_config_name}`",
            f"- Split: `{results.split_name}`",
            f"- Checkpoint: `{results.checkpoint_name}`",
            f"- Device: `{results.device}`",
            f"- Number of samples: {results.num_of_samples}",
            "",
            "## Correlation metrics:",
            f"- PLCC: `{results.correlation.plcc:.6f}`",
            f"- SRCC: `{results.correlation.srcc:.6f}`",
            f"- KRCC: `{results.correlation.krcc:.6f}`",
            "",
            "## Error metrics:",
            f"- MSE: {results.loss.mse:.6f}",
            f"- RMSE: {results.loss.rmse:.6f}",
            f"- MAE: {results.loss.mae:.6f}",
            ""
        ]

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as markdown_file:
            markdown_file.write('\n'.join(markdown_lines))
