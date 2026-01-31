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
    # 5-parametrowa funkcja logistyczna zgodna z podejściem Sheikha (IQA):
    #
    #     p(x, β) = β1 * ( 1/2 - 1 / (1 + exp( β2 * (x - β3) )) ) + β4 * x + β5
    #
    # gdzie:
    #   - x     : surowy wynik modelu / metryki jakości (predicted_scores)
    #   - β1    : skala (amplituda) części logistycznej
    #   - β2    : stromość (i znak monotoniczności) części logistycznej
    #   - β3    : przesunięcie w osi x (punkt "środka" sigmoidy)
    #   - β4    : człon liniowy (trend) – poprawia dopasowanie, gdy sama logistyka jest zbyt "sztywna"
    #   - β5    : przesunięcie w osi y (bias / offset)
    #
    # Uwaga numeryczna:
    #   exp(·) łatwo przepełnia się dla dużych argumentów, więc przycinamy argument wykładnika.
    exponent_argument = beta_2 * (predicted_scores - beta_3)
    exponent_argument = np.clip(exponent_argument, -60.0, 60.0)

    logistic_core = 0.5 - (1.0 / (1.0 + np.exp(exponent_argument)))
    return beta_1 * logistic_core + beta_4 * predicted_scores + beta_5


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

    # Typowe heurystyki startowe (dla funkcji Sheikha):
    #   - beta_1: skala części logistycznej (logistic_core ma zakres ~[-0.5, 0.5])
    #   - beta_2: stromość – rozsądny start to ~ 1 / std(predicted_scores)
    #   - beta_3: środek sigmoidy – startujemy od mediany predykcji
    #   - beta_4: trend liniowy – zwykle mały (0.0)
    #   - beta_5: offset – startujemy od średniej ground truth
    ground_truth_range = ground_truth_max - ground_truth_min

    initial_parameters = np.array(
        [
            2.0 * ground_truth_range if ground_truth_range > 0.0 else 1.0,  # beta_1
            1.0 / (predicted_standard_deviation + 1e-6),                    # beta_2
            predicted_median,                                               # beta_3
            0.0,                                                            # beta_4
            ground_truth_mean,                                              # beta_5
        ],
        dtype=np.float64,
    )

    # Ograniczenia - pomagają przy konwergencji i ograniczają "szalone" dopasowania.
    #
    # Celowo ustawiamy dość szerokie przedziały:
    #   - beta_1 może być dodatnie lub ujemne (odwrócenie skali), ale typowo jego rząd wielkości
    #     nie powinien być ogromny w porównaniu do rozpiętości ground truth.
    #   - beta_2 (stromość) może być dodatnie lub ujemne; duże wartości nadal są bezpieczne
    #     dzięki przycinaniu argumentu exp() w funkcji logistycznej.
    #   - beta_3 trzymamy w zakresie obserwowanych predykcji.
    #   - beta_4 to trend liniowy – szeroko, ale bez przesady.
    #   - beta_5 to przesunięcie (offset) – zakres oparty o ground truth.
    beta_1_abs_bound = max(10.0 * ground_truth_range, 1e-6)

    lower_bounds = np.array(
        [
            -beta_1_abs_bound,                                 # beta_1
            -200.0,                                            # beta_2
            predicted_min,                                     # beta_3
            -10.0,                                             # beta_4
            ground_truth_min - 2.0 * abs(ground_truth_min),     # beta_5
        ],
        dtype=np.float64,
    )

    upper_bounds = np.array(
        [
            beta_1_abs_bound,                                  # beta_1
            200.0,                                             # beta_2
            predicted_max,                                     # beta_3
            10.0,                                              # beta_4
            ground_truth_max + 2.0 * abs(ground_truth_max),    # beta_5
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
        predicted_scores: Iterable[float]
) -> CorrelationMetrics:
    ground_truth_array = np.asarray(ground_truth_scores, dtype=np.float64)
    predicted_array = np.asarray(predicted_scores, dtype=np.float64)

    if ground_truth_array.shape != predicted_array.shape:
        raise ValueError('Error: Tablica z etykietami i tablica z predykcjami muszą mieć ten sam rozmiar!')

    # PLCC
    predicted_array_for_plcc = _apply_nonlinear_regression(
        ground_truth_array,
        predicted_array
    )

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
