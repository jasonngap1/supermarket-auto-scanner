from app.pipeline.types import PromptSpec


def _clip_point(point, width, height):
    return (
        min(max(float(point[0]), 0.0), float(width)),
        min(max(float(point[1]), 0.0), float(height)),
    )


def clip_box(box, width, height):
    x1, y1, x2, y2 = box
    x1, y1 = _clip_point((x1, y1), width, height)
    x2, y2 = _clip_point((x2, y2), width, height)
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def build_box_prompt(box, frame_w: int, frame_h: int) -> PromptSpec:
    clipped = clip_box(box, frame_w, frame_h)
    x1, y1, x2, y2 = clipped
    if x2 <= x1 or y2 <= y1:
        return PromptSpec()
    centre = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return PromptSpec(positive_points=[centre], box=clipped)
