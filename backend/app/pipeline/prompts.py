import math

from app.pipeline.types import PromptSpec

NEGATIVE_INDICES = (0, 5, 9, 13, 17)


def _mid(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _mean(points):
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def _clip_point(point, width, height):
    return (
        min(max(point[0], 0.0), float(width - 1)),
        min(max(point[1], 0.0), float(height - 1)),
    )


def clip_box(box, width, height):
    x1, y1, x2, y2 = box
    x1, y1 = _clip_point((x1, y1), width, height)
    x2, y2 = _clip_point((x2, y2), width, height)
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def hand_bbox(landmarks):
    xs = [p[0] for p in landmarks]
    ys = [p[1] for p in landmarks]
    return (min(xs), min(ys), max(xs), max(ys))


def hand_scale(landmarks):
    x1, y1, x2, y2 = hand_bbox(landmarks)
    return max(x2 - x1, y2 - y1)


def _expand_box(box, margin, width, height):
    x1, y1, x2, y2 = box
    return clip_box((x1 - margin, y1 - margin, x2 + margin, y2 + margin), width, height)


def hand_crop_box(landmarks_px, frame_w, frame_h, expand=0.35):
    box = hand_bbox(landmarks_px)
    margin = expand * max(box[2] - box[0], box[3] - box[1])
    return _expand_box(box, margin, frame_w, frame_h)


def center_roi_box(
    frame_w: int, frame_h: int, width_ratio: float = 0.60, height_ratio: float = 0.70
):
    """Return a frame-centred search box for a held item."""
    width_ratio = min(max(width_ratio, 0.05), 1.0)
    height_ratio = min(max(height_ratio, 0.05), 1.0)
    half_w = frame_w * width_ratio / 2.0
    half_h = frame_h * height_ratio / 2.0
    cx, cy = frame_w / 2.0, frame_h / 2.0
    return clip_box((cx - half_w, cy - half_h, cx + half_w, cy + half_h), frame_w, frame_h)


def build_center_prompt(
    frame_w: int,
    frame_h: int,
    *,
    roi_width: float = 0.60,
    roi_height: float = 0.70,
    face_bbox: tuple[float, float, float, float] | None = None,
) -> PromptSpec:
    """Point + box prompt for an item held at the centre of the frame.

    The centre point is a hard object anchor; the box gives the segmenter the
    expected extent. Background corners and a visible face are negative
    prompts, so the segmenter prefers the item.
    """
    roi = center_roi_box(frame_w, frame_h, roi_width, roi_height)
    positive = (frame_w / 2.0, frame_h / 2.0)
    negatives = background_negative_points(frame_w, frame_h)
    negatives += box_negative_points(roi, frame_w, frame_h)
    if face_bbox is not None:
        negatives += face_negative_points(face_bbox, positive)
    return PromptSpec(
        positive_points=[positive],
        negative_points=list(dict.fromkeys(negatives)),
        box=roi,
        mode="sam2_center",
    )


def background_negative_points(
    frame_w: int, frame_h: int, margin_ratio: float = 0.04
) -> list[tuple[float, float]]:
    """Four frame corners, assumed to be background for a centred item."""
    mx, my = frame_w * margin_ratio, frame_h * margin_ratio
    return [
        (mx, my),
        (frame_w - 1 - mx, my),
        (mx, frame_h - 1 - my),
        (frame_w - 1 - mx, frame_h - 1 - my),
    ]


def box_negative_points(
    roi: tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
    margin_ratio: float = 0.02,
) -> list[tuple[float, float]]:
    """Corner points just outside the item box, assumed background."""
    mx, my = frame_w * margin_ratio, frame_h * margin_ratio
    x1, y1, x2, y2 = roi
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    points = [
        (x1 - mx, y1 - my),
        (x2 + mx, y1 - my),
        (x1 - mx, y2 + my),
        (x2 + mx, y2 + my),
        (cx, y1 - my),
        (cx, y2 + my),
        (x1 - mx, cy),
        (x2 + mx, cy),
    ]
    return [
        (min(max(x, 0.0), frame_w - 1.0), min(max(y, 0.0), frame_h - 1.0))
        for x, y in points
    ]


def face_negative_points(
    face_bbox: tuple[float, float, float, float],
    positive_point: tuple[float, float],
    inset_ratio: float = 0.18,
) -> list[tuple[float, float]]:
    """Negative points on a visible face.

    Skip them when the positive point is inside the face box: the item may be
    covering the face, and a negative point could land on the item.
    """
    x1, y1, x2, y2 = face_bbox
    px, py = positive_point
    if x1 <= px <= x2 and y1 <= py <= y2:
        return []
    inset_x, inset_y = (x2 - x1) * inset_ratio, (y2 - y1) * inset_ratio
    ax, bx = x1 + inset_x, x2 - inset_x
    ay, by = y1 + inset_y, y2 - inset_y
    return [
        ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        (ax, ay),
        (bx, ay),
        (ax, by),
        (bx, by),
    ]


def build_sam3_prompt(
    frame_w: int,
    frame_h: int,
    text: str,
    *,
    include_box: bool = True,
    roi_width: float = 0.60,
    roi_height: float = 0.70,
) -> PromptSpec:
    """Text concept prompt for SAM3, optionally anchored to a centred ROI box.

    SAM3 uses text concepts rather than hand-derived points. The box is only a
    localisation hint; the search region is also used to rank the returned
    instances.
    """
    roi = center_roi_box(frame_w, frame_h, roi_width, roi_height)
    return PromptSpec(
        positive_points=[],
        negative_points=[],
        box=roi if include_box else None,
        mode="sam3_center",
        text=text,
    )


def build_prompt(
    landmarks_px,
    frame_w,
    frame_h,
    *,
    mode="pinch_roi",
    pinch_close_ratio=0.35,
    box_expand=0.35,
) -> PromptSpec:
    if len(landmarks_px) != 21:
        raise ValueError("Expected 21 hand landmarks")

    landmarks = [_clip_point(p, frame_w, frame_h) for p in landmarks_px]
    palm_center = _mean([landmarks[i] for i in NEGATIVE_INDICES])
    pinch_center = _mid(landmarks[4], landmarks[8])
    grasp_center = _mid(landmarks[4], _mid(landmarks[8], landmarks[12]))
    scale = hand_scale(landmarks)
    pinch_distance = math.dist(landmarks[4], landmarks[8])
    positive = grasp_center if pinch_distance < pinch_close_ratio * scale else pinch_center
    negative = [landmarks[i] for i in NEGATIVE_INDICES] + [palm_center]
    negative = list(dict.fromkeys(negative))

    if mode == "none":
        box = None
    elif mode == "hand_expanded":
        box = hand_crop_box(landmarks, frame_w, frame_h, box_expand)
    elif mode == "pinch_roi":
        roi = hand_bbox([landmarks[4], landmarks[8], landmarks[12], palm_center])
        box = _expand_box(roi, box_expand * scale, frame_w, frame_h)
    else:
        raise ValueError(f"Unknown prompt mode: {mode}")

    return PromptSpec(
        positive_points=[positive],
        negative_points=negative,
        box=box,
        mode=mode,
    )
