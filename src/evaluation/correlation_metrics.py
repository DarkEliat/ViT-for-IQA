from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class CorrelationMetrics:
    plcc: float
    srcc: float
    krcc: float


def _five_parameter_logistic_function(
        x: np.ndarray,
        beta_1: float,
        beta_2: float,
        beta_3: float,
        beta_4: float,
        beta_5: float,
) -> np.ndarray:
    return (
        beta_1 * (
            0.5 - 1.0 / (
                1.0 + np.exp(
                    beta_2 * (x - beta_3)
                )
            )
        )
        + beta_4 * x
        + beta_5
    )


def _apply_nonlinear_regression(
        predicted_scores: np.ndarray,
        ground_truth_scores: np.ndarray
) -> np.ndarray:
    initial_parameters = np.array([1.0, 1.0, 0.5, 0.0, 0.0])

    optimized_parameters, covariance_matrix, _, _, _ = curve_fit(
        _five_parameter_logistic_function,
        predicted_scores,
        ground_truth_scores,
        p0=initial_parameters,
        maxfev=10000,
        full_output=True
    )

    return _five_parameter_logistic_function(
        predicted_scores,
        *optimized_parameters
    )


def compute_correlations(
        ground_truth_scores: Iterable[float],
        predicted_scores: Iterable[float],
        apply_nonlinear_regression_for_plcc: bool = True
) -> CorrelationMetrics:
    ground_truth_array = np.asarray(ground_truth_scores, dtype=np.float64)
    predicted_array = np.asarray(predicted_scores, dtype=np.float64)

    if ground_truth_array.shape != predicted_array.shape:
        raise ValueError('Error: Tablica z etykietami i tablica z predykcjami muszą mieć ten sam rozmiar!')

    # PLCC
    if apply_nonlinear_regression_for_plcc:
        predicted_array_for_plcc = _apply_nonlinear_regression(
            predicted_array,
            ground_truth_array
        )
    else:
        predicted_array_for_plcc = predicted_array

    plcc_value, _ = pearsonr(predicted_array_for_plcc, ground_truth_array)

    # SRCC
    srcc_value, _ = spearmanr(predicted_array.tolist(), ground_truth_array.tolist())

    # KRCC
    krcc_value, _ = kendalltau(predicted_array.tolist(), ground_truth_array.tolist())

    return CorrelationMetrics(
        plcc=float(plcc_value),
        srcc=float(srcc_value),
        krcc=float(krcc_value)
    )
