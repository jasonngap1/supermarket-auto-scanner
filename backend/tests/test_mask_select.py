import numpy as np

from app.pipeline.mask_select import mask_bbox, select_center_mask
from app.pipeline.types import MaskCandidate


def make_mask(x1, y1, x2, y2, size=40):
    mask = np.zeros((size, size), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def center_select(candidates, **kwargs):
    return select_center_mask(
        candidates,
        (50, 50),
        (20, 15, 80, 85),
        100,
        100,
        min_area_ratio=0.005,
        max_area_ratio=0.60,
        min_pred_iou=0.5,
        **kwargs,
    )


def test_bbox_helper():
    assert mask_bbox(make_mask(3, 4, 8, 9)) == (3, 4, 8, 9)
    assert mask_bbox(np.zeros((5, 5), dtype=bool)) is None


def test_center_rejects_face_and_border_masks():
    item = MaskCandidate(0, make_mask(40, 40, 60, 60, size=100), 0.7)
    face = MaskCandidate(1, make_mask(35, 35, 65, 65, size=100), 0.95)
    result = center_select([item, face], face_bbox=(35, 35, 65, 65))
    assert result.selected is not None
    assert result.selected.index == 0
    assert result.reasons[1] == "face_mask"

    border = np.zeros((100, 100), dtype=bool)
    border[:3, :] = border[-3:, :] = True
    border[:, :3] = border[:, -3:] = True
    border[45:55, 45:55] = True
    ring = MaskCandidate(2, border, 0.99)
    result = select_center_mask(
        [ring],
        (50, 50),
        None,
        100,
        100,
        min_area_ratio=0.005,
        max_area_ratio=0.60,
        min_pred_iou=0.5,
    )
    assert result.selected is None
    assert result.reasons[0] == "touches_border"


def test_center_rejects_mask_outside_the_centre_roi():
    outside = MaskCandidate(0, make_mask(0, 40, 60, 60, size=100), 0.9)
    result = center_select([outside])
    assert result.selected is None
    assert result.reasons[0] == "outside_roi"


def test_center_rejects_low_confidence_and_small_masks():
    low = MaskCandidate(0, make_mask(40, 40, 60, 60, size=100), 0.2)
    tiny = MaskCandidate(1, make_mask(50, 50, 51, 51, size=100), 0.9)
    result = center_select([low, tiny])
    assert result.selected is None
    assert result.reasons == ["low_iou", "area_too_small"]
