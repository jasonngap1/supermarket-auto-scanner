import math

import numpy as np

from app.pipeline.types import MaskCandidate, MaskSelection


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def point_inside(mask: np.ndarray, point: tuple[float, float], pad: int = 2) -> bool:
    height, width = mask.shape[:2]
    x = int(round(point[0]))
    y = int(round(point[1]))
    x1, x2 = max(0, x - pad), min(width, x + pad + 1)
    y1, y2 = max(0, y - pad), min(height, y + pad + 1)
    if x1 >= x2 or y1 >= y2:
        return False
    return bool(mask[y1:y2, x1:x2].any())


def bbox_overlaps(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


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


def select_center_mask(
    candidates: list[MaskCandidate],
    positive_point,
    search_roi,
    frame_w: int,
    frame_h: int,
    *,
    face_bbox=None,
    min_area_ratio: float,
    max_area_ratio: float,
    min_pred_iou: float = 0.0,
    max_border_fraction: float = 0.08,
    face_iou_reject: float = 0.75,
    face_coverage_reject: float = 0.70,
    roi_margin_ratio: float = 0.03,
    center_weight: float = 0.15,
) -> MaskSelection:
    """Pick a centre-held item mask and reject face/border/background masks."""
    reasons: list[str] = []
    scores: list[float] = []
    areas: list[float] = []
    frame_area = float(frame_w * frame_h) if frame_w and frame_h else 1.0

    expanded_roi = None
    if search_roi is not None:
        margin_x = frame_w * roi_margin_ratio
        margin_y = frame_h * roi_margin_ratio
        expanded_roi = (
            search_roi[0] - margin_x,
            search_roi[1] - margin_y,
            search_roi[2] + margin_x,
            search_roi[3] + margin_y,
        )

    positive_in_face = False
    if face_bbox is not None:
        px, py = positive_point
        positive_in_face = (
            face_bbox[0] <= px <= face_bbox[2]
            and face_bbox[1] <= py <= face_bbox[3]
        )

    for candidate in candidates:
        mask = candidate.mask
        box = mask_bbox(mask)
        area_ratio = float(mask.sum()) / frame_area
        areas.append(area_ratio)

        border_fraction = _border_fraction(mask)
        outside_roi = box is not None and expanded_roi is not None and (
            box[0] < expanded_roi[0]
            or box[1] < expanded_roi[1]
            or box[2] > expanded_roi[2]
            or box[3] > expanded_roi[3]
        )
        face_iou = (
            _rect_iou(box, face_bbox)
            if box is not None and face_bbox is not None
            else 0.0
        )
        face_coverage = 0.0
        if face_bbox is not None:
            face_area = _rect_area(face_bbox)
            if face_area > 0:
                face_coverage = _rect_intersection_area(mask, face_bbox) / face_area

        if box is None:
            reason = "empty"
        elif candidate.pred_iou < min_pred_iou:
            reason = "low_iou"
        elif not point_inside(mask, positive_point):
            reason = "positive_outside"
        elif area_ratio < min_area_ratio:
            reason = "area_too_small"
        elif area_ratio > max_area_ratio:
            reason = "area_too_large"
        elif search_roi is not None and not bbox_overlaps(box, search_roi):
            reason = "no_roi_overlap"
        elif outside_roi:
            reason = "outside_roi"
        elif border_fraction > max_border_fraction:
            reason = "touches_border"
        elif face_iou >= face_iou_reject or (
            not positive_in_face and face_coverage >= face_coverage_reject
        ):
            reason = "face_mask"
        else:
            reason = "ok"

        if box is None:
            center_distance = 1.0
        else:
            center_distance = _center_distance(box, frame_w, frame_h)
        area_penalty = 0.05 * max(0.0, area_ratio - 0.15) / 0.15
        border_penalty = 0.10 if border_fraction > 0 else 0.0
        outside_penalty = 0.10 if outside_roi else 0.0
        face_penalty = 0.20 if face_iou > 0.15 or face_coverage > 0.15 else 0.0
        score = (
            float(candidate.pred_iou)
            - center_weight * center_distance
            - area_penalty
            - border_penalty
            - outside_penalty
            - face_penalty
        )
        reasons.append(reason)
        scores.append(score)

    eligible = [
        (scores[i], areas[i], candidates[i])
        for i, reason in enumerate(reasons)
        if reason == "ok"
    ]
    selected = max(eligible, key=lambda item: (item[0], item[1]))[2] if eligible else None
    return MaskSelection(selected=selected, reasons=reasons, scores=scores, area_ratios=areas)


def _center_distance(box, frame_w: int, frame_h: int) -> float:
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    dx = (cx - frame_w / 2.0) / (frame_w / 2.0) if frame_w else 0.0
    dy = (cy - frame_h / 2.0) / (frame_h / 2.0) if frame_h else 0.0
    return min(1.0, math.hypot(dx, dy) / math.sqrt(2.0))
