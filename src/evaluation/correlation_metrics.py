from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.optimize import curve_fit

from src.utils.data_types import CorrelationMetrics


def _five_parameter_logistic_function(
    predicted_scores: np.ndarray,
    beta_1: float,
    beta_2: float,
    beta_3: float,
    beta_4: float,
    beta_5: float,
) -> np.ndarray:
    # Klasyczna postać spotykana w pracach IQA / benchmarkach.
    # Zapis poniżej jest stabilny numerycznie dla typowych zakresów danych.
    return beta_2 + (beta_1 - beta_2) / (1.0 + np.exp(-(predicted_scores - beta_3) / (np.abs(beta_4) + 1e-12))) + beta_5 * predicted_scores


def _apply_nonlinear_regression(
    ground_truth_scores: np.ndarray,
    predicted_scores: np.ndarray,
    max_number_of_function_evaluations: int = 200_000,
) -> np.ndarray:
    ground_truth_scores = np.asarray(ground_truth_scores, dtype=np.float64).ravel()
    predicted_scores = np.asarray(predicted_scores, dtype=np.float64).ravel()

    # Walidacja podstawowa
    if ground_truth_scores.size != predicted_scores.size:
        raise ValueError(
            "Error: `ground_truth_scores` i `predicted_scores` mają różne długości.\n"
            f"    ground_truth_scores.size={ground_truth_scores.size}\n"
            f"    predicted_scores.size={predicted_scores.size}"
        )

    if ground_truth_scores.size < 8:
        # Przy bardzo małej liczbie próbek dopasowanie 5 parametrów jest niestabilne.
        return predicted_scores

    if not np.all(np.isfinite(ground_truth_scores)) or not np.all(np.isfinite(predicted_scores)):
        # Jeżeli coś jest NaN/Inf, regresja i tak się nie powiedzie.
        # Lepiej wrócić do surowych predykcji.
        return predicted_scores

    # Przypadek prawie stałych predykcji
    predicted_standard_deviation = float(np.std(predicted_scores))
    if predicted_standard_deviation < 1e-8:
        # Gdy model przewiduje prawie stałą, logistyczna nie ma sensu i często nie konwerguje.
        return predicted_scores

    # Wartości startowe (p0)
    # Ground truth (MOS/DMOS) – przydatne do ustawienia skali
    ground_truth_min = float(np.min(ground_truth_scores))
    ground_truth_max = float(np.max(ground_truth_scores))
    ground_truth_mean = float(np.mean(ground_truth_scores))

    predicted_min = float(np.min(predicted_scores))
    predicted_max = float(np.max(predicted_scores))
    predicted_median = float(np.median(predicted_scores))

    # Typowe heurystyki startowe:
    # beta_1, beta_2 ustawiają początek i koniec skali ground-truth,
    # beta_3 to przesunięcie (środek), beta_4 to "szerokość", beta_5 to mały trend liniowy
    initial_parameters = np.array(
        [
            ground_truth_max,                               # beta_1
            ground_truth_min,                               # beta_2
            predicted_median,                               # beta_3
            (predicted_max - predicted_min) / 4.0 + 1e-6,   # beta_4
            0.0,                                            # beta_5
        ],
        dtype=np.float64,
    )

    # Ograniczenia - pomagają przy konwergencji
    lower_bounds = np.array(
        [
            ground_truth_min - 2.0 * abs(ground_truth_min),  # beta_1
            ground_truth_min - 2.0 * abs(ground_truth_min),  # beta_2
            predicted_min,                                   # beta_3
            1e-8,                                            # beta_4 (musi być dodatni, bo stoi w mianowniku)
            -10.0,                                           # beta_5
        ],
        dtype=np.float64,
    )

    upper_bounds = np.array(
        [
            ground_truth_max + 2.0 * abs(ground_truth_max),  # beta_1
            ground_truth_max + 2.0 * abs(ground_truth_max),  # beta_2
            predicted_max,                                   # beta_3
            (predicted_max - predicted_min) * 10.0 + 1e-6,   # beta_4
            10.0,                                            # beta_5
        ],
        dtype=np.float64,
    )

    # Dopasowanie
    try:
        optimized_parameters, _covariance_matrix, _, _, _ = curve_fit(
            f=_five_parameter_logistic_function,
            xdata=predicted_scores,
            ydata=ground_truth_scores,
            p0=initial_parameters,
            bounds=(lower_bounds, upper_bounds),
            maxfev=max_number_of_function_evaluations,
            full_output=True
        )
    except Exception:
        return predicted_scores

    mapped_scores = _five_parameter_logistic_function(predicted_scores, *optimized_parameters)
    return mapped_scores


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
            ground_truth_array,
            predicted_array
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
