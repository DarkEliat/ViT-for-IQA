from typing import TypedDict, Any, Literal
from dataclasses import dataclass


from torch import Tensor


Config = dict[str, Any]


class QualityScore(TypedDict):
    type: str
    min_value: float
    max_value: float
    value: float
    normalized: bool
    model_target: bool


@dataclass(frozen=True)
class UnifiedQualityScore:
    value: float
    min_value: float = 0.0
    max_value: float = 1.0
    type: str = 'unified'
    normalized: bool = True
    model_target: bool = True

    def __post_init__(self) -> None:
        if not (self.min_value <= self.value <= self.max_value):
            raise ValueError('Error: Ustawiona wartość zunifikowanego wskaźnika jakości `UnifiedQualityScore` wykracza poza zakres <0.0, 1.0>!')



class Label(TypedDict):
    reference_image_name: str
    distorted_image_name: str
    quality_score: QualityScore


StateDict = dict[str, Tensor]

class Checkpoint(TypedDict):
    epoch: int
    model_state_dict: StateDict
    optimizer_state_dict: StateDict
    train_loss: float
    validation_loss: float


SplitName = Literal['train', 'validation', 'test']

@dataclass(frozen=True)
class EvaluationResults:
    plcc: float
    srcc: float
    krcc: float
    mse: float
    rmse: float
    mae: float
    num_of_samples: int
    split_name: SplitName
    checkpoint_name: str
    config_name: str
    dataset_name: str
    device: str
