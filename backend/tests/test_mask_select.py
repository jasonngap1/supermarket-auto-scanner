import numpy as np

from app.pipeline.mask_select import select_box_mask
from app.pipeline.types import MaskCandidate

BOX = (30, 30, 70, 70)


def candidate(mask, pred_iou=0.9, index=0):
    return MaskCandidate(index, mask, pred_iou)


def mask_for(box):
    mask = np.zeros((100, 100), dtype=bool)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = True
    return mask


def select(candidates, yolo_box=BOX, **kwargs):
    options = {"min_area_ratio": 0.005, "max_area_ratio": 0.60}
    options.update(kwargs)
    return select_box_mask(candidates, yolo_box, 100, 100, **options)


def test_matching_mask_is_selected():
    result = select([candidate(mask_for(BOX))])
    assert result.selected is not None
    assert result.reasons == ["ok"]


def test_low_pred_iou_is_rejected():
    result = select([candidate(mask_for(BOX), pred_iou=0.3)])
    assert result.selected is None
    assert result.reasons == ["low_pred_iou"]


def test_mask_outside_the_box_is_rejected():
    result = select([candidate(mask_for((20, 30, 80, 70)))])
    assert result.selected is None
    assert result.reasons == ["outside_box"]


def test_mask_with_low_box_iou_is_rejected():
    result = select([candidate(mask_for((5, 5, 25, 25)))])
    assert result.selected is None
    assert result.reasons == ["low_box_iou"]


def test_small_mask_is_rejected():
    result = select([candidate(mask_for((50, 50, 51, 51)))])
    assert result.selected is None
    assert result.reasons == ["too_small"]


def test_large_mask_is_rejected():
    result = select([candidate(mask_for((10, 10, 90, 90)))])
    assert result.selected is None
    assert result.reasons == ["too_large"]


def test_border_mask_is_rejected():
    ring = np.zeros((100, 100), dtype=bool)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    result = select(
        [candidate(ring)],
        yolo_box=(0, 0, 100, 100),
        max_area_ratio=1.1,
    )
    assert result.selected is None
    assert result.reasons == ["border"]


def test_highest_scoring_mask_wins():
    result = select(
        [
            candidate(mask_for(BOX), pred_iou=0.6, index=0),
            candidate(mask_for(BOX), pred_iou=0.9, index=1),
        ]
    )
    assert result.selected.index == 1
