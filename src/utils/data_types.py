from typing import TypedDict


class QualityScore(TypedDict):
    type: str
    value: float
    normalized: bool
    model_target: bool


class Label(TypedDict):
    reference_image_name: str
    distorted_image_name: str
    quality_score: QualityScore
