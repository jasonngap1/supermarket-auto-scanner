from app.pipeline.types import PromptSpec


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
    """Point + box prompt for an item held at the centre of the frame."""
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
