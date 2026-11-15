import math

import numpy as np

from app.pipeline.types import Detection, DetectionSelection, MaskCandidate, MaskSelection


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _rect_area(rect) -> float:
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def _rect_intersection_area(mask: np.ndarray, rect) -> int:
    height, width = mask.shape[:2]
    x1 = max(0, min(width, int(round(rect[0]))))
    y1 = max(0, min(height, int(round(rect[1]))))
    x2 = max(0, min(width, int(round(rect[2]))))
    y2 = max(0, min(height, int(round(rect[3]))))
    if x1 >= x2 or y1 >= y2:
        return 0
    return int(mask[y1:y2, x1:x2].sum())


def _rect_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _rect_area(a) + _rect_area(b) - inter
    return inter / union if union > 0 else 0.0


def _border_fraction(mask: np.ndarray) -> float:
    pixels = int(mask.sum())
    if pixels == 0:
        return 0.0
    height, width = mask.shape[:2]
    count = (
        int(mask[0, :].sum())
        + int(mask[-1, :].sum())
        + int(mask[:, 0].sum())
        + int(mask[:, -1].sum())
    )
    if height > 1 and width > 1:
        count -= int(mask[0, 0])
        count -= int(mask[0, -1])
        count -= int(mask[-1, 0])
        count -= int(mask[-1, -1])
    return count / pixels


def _center_distance(box, frame_w: int, frame_h: int) -> float:
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    dx = (cx - frame_w / 2.0) / (frame_w / 2.0) if frame_w else 0.0
    dy = (cy - frame_h / 2.0) / (frame_h / 2.0) if frame_h else 0.0
    return min(1.0, math.hypot(dx, dy) / math.sqrt(2.0))


def _touches_border(box, frame_w: int, frame_h: int, margin: int = 2) -> bool:
    return (
        box[0] <= margin
        or box[1] <= margin
        or box[2] >= frame_w - margin
        or box[3] >= frame_h - margin
    )


def select_detection(
    detections: list[Detection],
    frame_w: int,
    frame_h: int,
    *,
    min_area_ratio: float,
    max_area_ratio: float,
    reject_classes: frozenset[str] = frozenset(),
    center_weight: float = 0.15,
    min_confidence: float = 0.0,
) -> DetectionSelection:
    frame_area = float(frame_w * frame_h)
    best: Detection | None = None
    best_score = float("-inf")
    reasons: list[str] = []
    scores: list[float] = []
    area_ratios: list[float] = []

    for detection in detections:
        area = _rect_area(detection.bbox)
        ratio = area / frame_area if frame_area else 0.0
        reason = "ok"
        score = float("-inf")

        if area <= 0:
            reason = "empty_bbox"
        elif detection.confidence < min_confidence:
            reason = "low_confidence"
        elif detection.class_name in reject_classes:
            reason = "rejected_class"
        elif ratio < min_area_ratio:
            reason = "too_small"
        elif ratio > max_area_ratio:
            reason = "too_large"
        elif _touches_border(detection.bbox, frame_w, frame_h):
            reason = "border"
        else:
            score = detection.confidence - center_weight * _center_distance(
                detection.bbox, frame_w, frame_h
            )
            if score > best_score:
                best, best_score = detection, score

        reasons.append(reason)
        scores.append(score)
        area_ratios.append(ratio)

    return DetectionSelection(best, reasons, scores, area_ratios)


def select_box_mask(
    candidates: list[MaskCandidate],
    yolo_box: tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
    *,
    min_area_ratio: float,
    max_area_ratio: float,
    min_pred_iou: float = 0.50,
    min_box_iou: float = 0.40,
    min_box_coverage: float = 0.75,
    max_border_fraction: float = 0.10,
    box_weight: float = 0.50,
    area_weight: float = 0.25,
    border_weight: float = 0.10,
) -> MaskSelection:
    frame_area = float(frame_w * frame_h)
    box_area = max(1.0, _rect_area(yolo_box))
    best: MaskCandidate | None = None
    best_score = float("-inf")
    reasons: list[str] = []
    scores: list[float] = []
    area_ratios: list[float] = []

    for candidate in candidates:
        box = mask_bbox(candidate.mask)
        pixels = int(candidate.mask.sum())
        ratio = pixels / frame_area if frame_area else 0.0
        reason = "ok"
        score = float("-inf")

        if box is None or pixels == 0:
            reason = "empty"
        elif candidate.pred_iou < min_pred_iou:
            reason = "low_pred_iou"
        elif ratio < min_area_ratio:
            reason = "too_small"
        elif ratio > max_area_ratio:
            reason = "too_large"
        else:
            border = _border_fraction(candidate.mask)
            box_iou = _rect_iou(box, yolo_box)
            coverage = _rect_intersection_area(candidate.mask, yolo_box) / pixels
            if border > max_border_fraction:
                reason = "border"
            elif box_iou < min_box_iou:
                reason = "low_box_iou"
            elif coverage < min_box_coverage:
                reason = "outside_box"
            else:
                area_mismatch = min(1.0, abs(pixels - box_area) / box_area)
                score = (
                    candidate.pred_iou
                    + box_weight * box_iou
                    - area_weight * area_mismatch
                    - border_weight * border
                )
                if score > best_score:
                    best, best_score = candidate, score

        reasons.append(reason)
        scores.append(score)
        area_ratios.append(ratio)

    return MaskSelection(best, reasons, scores, area_ratios)
