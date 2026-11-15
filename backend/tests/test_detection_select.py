from app.pipeline.mask_select import select_detection
from app.pipeline.types import Detection


def detection(index, bbox, confidence=0.9, class_name="banana"):
    return Detection(index, 0, class_name, confidence, bbox)


def select(detections, **kwargs):
    return select_detection(
        detections,
        100,
        100,
        min_area_ratio=0.005,
        max_area_ratio=0.60,
        **kwargs,
    )


def test_centre_detection_wins_when_scores_tie():
    result = select(
        [
            detection(0, (5, 5, 25, 25)),
            detection(1, (35, 35, 65, 65)),
        ]
    )
    assert result.selected.index == 1


def test_rejected_class_is_skipped():
    result = select(
        [
            detection(0, (35, 35, 65, 65), class_name="person"),
            detection(1, (5, 5, 25, 25)),
        ],
        reject_classes=frozenset({"person"}),
    )
    assert result.selected.index == 1
    assert result.reasons[0] == "rejected_class"


def test_border_detection_is_rejected():
    result = select([detection(0, (0, 20, 40, 80))])
    assert result.selected is None
    assert result.reasons[0] == "border"


def test_area_limits_are_enforced():
    result = select(
        [
            detection(0, (10, 10, 12, 12)),
            detection(1, (0, 0, 99, 99)),
        ]
    )
    assert result.selected is None
    assert result.reasons == ["too_small", "too_large"]


def test_low_confidence_is_rejected():
    result = select(
        [detection(0, (30, 30, 70, 70), confidence=0.1)],
        min_confidence=0.25,
    )
    assert result.selected is None
    assert result.reasons[0] == "low_confidence"


def test_highest_confidence_wins_among_eligible_detections():
    result = select(
        [
            detection(0, (5, 5, 25, 25), confidence=0.5),
            detection(1, (35, 35, 65, 65), confidence=0.9),
        ]
    )
    assert result.selected.index == 1
