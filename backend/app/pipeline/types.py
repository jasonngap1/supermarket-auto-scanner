from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class FaceObservation:
    bbox: tuple[float, float, float, float]
    score: float


@dataclass(frozen=True)
class PromptSpec:
    positive_points: list[tuple[float, float]]
    negative_points: list[tuple[float, float]]
    box: tuple[float, float, float, float] | None
    mode: str
    text: str | None = None


@dataclass(frozen=True)
class MaskCandidate:
    index: int
    mask: np.ndarray
    pred_iou: float


@dataclass(frozen=True)
class MaskSelection:
    selected: MaskCandidate | None
    reasons: list[str]
    scores: list[float]
    area_ratios: list[float]


@dataclass(frozen=True)
class Localization:
    prompt: PromptSpec | None
    selection: MaskSelection
    mask: np.ndarray | None
    status: str  # ok | mask_rejected | model_error
    timings_ms: dict[str, float] = field(default_factory=dict)
    face: FaceObservation | None = None
