from typing import TypedDict, Any, Literal
from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class CorrelationMetrics:
    plcc: float = float('-inf')
    srcc: float = float('-inf')
    krcc: float = float('-inf')


@dataclass(frozen=True)
class LossMetrics:
    mse: float = float('inf')
    rmse: float = float('inf')
    mae: float = float('inf')


SplitName = Literal['train', 'validation', 'test']

@dataclass(frozen=True)
class EvaluationResults:
    correlation: CorrelationMetrics
    loss: LossMetrics
    num_of_samples: int
    split_name: SplitName
    checkpoint_name: str
    config_name: str
    dataset_name: str
    device: str


@dataclass
class CheckpointInfo:
    epoch: int = field(default=0)
    train_loss: LossMetrics = field(default_factory=LossMetrics)
    validation_loss: LossMetrics = field(default_factory=LossMetrics)
    validation_correlation: CorrelationMetrics = field(default_factory=CorrelationMetrics)


StateDict = dict[str, Tensor]

@dataclass(frozen=True)
class CheckpointPickle:
    model_state_dict: StateDict
    optimizer_state_dict: StateDict
    last_epoch: CheckpointInfo
    best_epoch: CheckpointInfo
    config: Config
