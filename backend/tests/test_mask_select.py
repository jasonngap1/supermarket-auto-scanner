import numpy as np

from app.pipeline.mask_select import (
    mask_bbox,
    select_center_mask,
    select_mask,
    select_sam3_mask,
)
from app.pipeline.types import MaskCandidate


def make_mask(x1, y1, x2, y2, size=40):
    mask = np.zeros((size, size), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def run(candidates, landmarks=None, positive=(20, 20), negatives=((5, 5),), roi=(0, 0, 40, 40)):
    return select_mask(
        candidates,
        landmarks or [(5, 5)] * 21,
        positive,
        list(negatives),
        roi,
        frame_area=40 * 40,
        min_area_ratio=0.003,
        max_area_ratio=0.30,
        max_landmark_hits=6,
    )


def test_selects_mask_containing_positive_and_avoiding_negatives():
    good = MaskCandidate(0, make_mask(10, 10, 30, 30), 0.9)
    hand = MaskCandidate(1, make_mask(0, 0, 40, 40), 0.99)
    result = run([hand, good], negatives=((5, 5),))
    assert result.selected is not None
    assert result.selected.index == 0
    assert "negative_inside" in result.reasons[0]


def test_rejects_positive_outside_and_too_small():
    outside = MaskCandidate(0, make_mask(0, 0, 10, 10), 0.9)
    tiny = MaskCandidate(1, make_mask(20, 20, 22, 22), 0.95)
    result = run([outside, tiny], positive=(21, 21))
    assert result.selected is None
    assert result.reasons[0] == "positive_outside"
    assert result.reasons[1] == "area_too_small"


def test_rejects_too_many_landmarks():
    big = MaskCandidate(0, make_mask(0, 0, 30, 30), 0.9)
    landmarks = [(5, 5)] * 7 + [(20, 20)] * 14
    result = run([big], landmarks=landmarks, positive=(20, 20), negatives=((39, 39),))
    assert result.selected is None
    assert result.reasons[0] == "too_many_landmarks"


def test_rejects_low_confidence_candidate():
    low = MaskCandidate(0, make_mask(10, 10, 30, 30), 0.5)
    result = select_mask(
        [low],
        [(5, 5)] * 21,
        (20, 20),
        [(5, 5)],
        (0, 0, 40, 40),
        frame_area=40 * 40,
        min_area_ratio=0.003,
        max_area_ratio=0.30,
        max_landmark_hits=6,
        min_pred_iou=0.55,
    )
    assert result.selected is None
    assert result.reasons[0] == "low_iou"


def test_bbox_helper():
    assert mask_bbox(make_mask(3, 4, 8, 9)) == (3, 4, 8, 9)
    assert mask_bbox(np.zeros((5, 5), dtype=bool)) is None


def test_sam3_prefers_the_centred_instance():
    corner = MaskCandidate(1, make_mask(0, 0, 20, 20), 0.85)
    centre = MaskCandidate(0, make_mask(10, 10, 30, 30), 0.80)
    result = select_sam3_mask(
        [corner, centre],
        (5, 5, 35, 35),
        40,
        40,
        min_area_ratio=0.003,
        max_area_ratio=0.30,
    )
    assert result.selected is not None
    assert result.selected.index == 0


def test_sam3_rejects_low_score_and_masks_outside_the_roi():
    low = MaskCandidate(0, make_mask(10, 10, 30, 30), 0.2)
    outside = MaskCandidate(1, make_mask(0, 0, 5, 5), 0.9)
    result = select_sam3_mask(
        [low, outside],
        (10, 10, 30, 30),
        40,
        40,
        min_area_ratio=0.003,
        max_area_ratio=0.30,
        min_score=0.3,
    )
    assert result.selected is None
    assert result.reasons[0] == "low_score"
    assert result.reasons[1] == "no_roi_overlap"


def test_sam3_rejects_empty_and_oversized_masks():
    empty = MaskCandidate(0, np.zeros((40, 40), dtype=bool), 0.9)
    huge = MaskCandidate(1, np.ones((40, 40), dtype=bool), 0.9)
    result = select_sam3_mask(
        [empty, huge],
        (0, 0, 40, 40),
        40,
        40,
        min_area_ratio=0.003,
        max_area_ratio=0.30,
    )
    assert result.selected is None
    assert result.reasons == ["empty", "area_too_large"]


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
