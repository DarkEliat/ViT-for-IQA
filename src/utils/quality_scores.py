from src.utils.data_types import UnifiedQualityScore


def normalize_min_max(
    score_value: float,
    score_min: float,
    score_max: float
) -> float:
    if score_min >= score_max:
        raise ValueError('Error: Wartość `score_min` musi być mniejsza niż `score_max`!')

    if not score_min <= score_value <= score_max:
        raise ValueError(f"Error: Wartość 'score_value' wykracza poza przekazany dopuszczalny zakres: <{score_min}, {score_max}>!")

    normalized_score = (score_value - score_min) / (score_max - score_min)

    return normalized_score


def dmos_to_quality_score(
        dmos_value: float,
        dmos_min: float = 0.0,
        dmos_max: float = 0.0
) -> UnifiedQualityScore:
    normalized_dmos_value = normalize_min_max(score_value=dmos_value, score_min=dmos_min, score_max=dmos_max)

    quality_score_value = 1.0 - normalized_dmos_value

    return UnifiedQualityScore(value=quality_score_value)
