from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PromptSpec:
    positive_points: list[tuple[float, float]] = field(default_factory=list)
    box: tuple[float, float, float, float] | None = None


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
class Detection:
    index: int
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    # Boolean HxW mask at the original frame size. compare/repr are off so
    # equality and logging never touch the array.
    mask: np.ndarray | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class DetectionSelection:
    selected: Detection | None
    reasons: list[str]
    scores: list[float]
    area_ratios: list[float]


@dataclass(frozen=True)
class Localization:
    status: str = "ok"  # ok | no_detection | mask_rejected | model_error
    mask: np.ndarray | None = None
    bbox: tuple[float, float, float, float] | None = None
    prompt: PromptSpec | None = None
    mask_selection: MaskSelection | None = None
    detections: list[Detection] = field(default_factory=list)
    detection_selection: DetectionSelection | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
