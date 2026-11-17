from types import SimpleNamespace

import numpy as np
import pytest

from app.errors import ModelUnavailable
from app.pipeline.yolo import (
    detect_and_select,
    localize_with_yolo_seg,
    masks_from_result,
    parse_class_ids,
)
from tests.fakes import FakeRegistry, detection, image, make_settings


def test_parse_class_ids_accepts_names_and_ids():
    names = {0: "person", 46: "banana"}
    assert parse_class_ids("banana,0", names) == [46, 0]
    assert parse_class_ids("", names) is None


def test_parse_class_ids_rejects_unknown_names():
    with pytest.raises(ModelUnavailable):
        parse_class_ids("unicorn", {0: "person"})


def test_masks_from_result_converts_uint8_to_bool():
    data = np.zeros((1, 10, 10), dtype=np.uint8)
    data[0, 2:6, 3:7] = 1
    result = SimpleNamespace(masks=SimpleNamespace(data=data))

    masks = masks_from_result(result, 1, (10, 10))

    assert masks[0] is not None
    assert masks[0].dtype == bool
    assert int(masks[0].sum()) == 16


def test_masks_from_result_handles_missing_and_mismatched_masks():
    assert masks_from_result(SimpleNamespace(masks=None), 2) == [None, None]
    data = np.zeros((1, 10, 10), dtype=np.uint8)
    result = SimpleNamespace(masks=SimpleNamespace(data=data))
    assert masks_from_result(result, 2) == [None, None]


def test_detection_mask_is_used_directly(tmp_path):
    registry = FakeRegistry(detections=[detection()])
    localization = localize_with_yolo_seg(
        make_settings(tmp_path), registry, image()
    )
    assert localization.status == "ok"
    assert localization.bbox == (20.0, 20.0, 80.0, 80.0)
    assert localization.mask is not None and localization.mask.sum() == 3600
    assert localization.timings_ms.keys() == {"yolo"}


def test_no_detection_reports_no_detection(tmp_path):
    localization = localize_with_yolo_seg(
        make_settings(tmp_path), FakeRegistry(detections=[]), image()
    )
    assert localization.status == "no_detection"
    assert localization.mask is None


def test_rejected_class_reports_no_detection(tmp_path):
    registry = FakeRegistry(
        detections=[detection(class_id=0, class_name="person")]
    )
    localization = localize_with_yolo_seg(
        make_settings(tmp_path), registry, image()
    )
    assert localization.status == "no_detection"
    assert localization.detection_selection.reasons == ["rejected_class"]


def test_detection_without_mask_reports_mask_rejected(tmp_path):
    registry = FakeRegistry(detections=[detection(with_mask=False)])
    localization = localize_with_yolo_seg(
        make_settings(tmp_path), registry, image()
    )
    assert localization.status == "mask_rejected"
    assert localization.bbox == (20.0, 20.0, 80.0, 80.0)
    assert localization.mask is None


def test_detector_failure_propagates(tmp_path):
    registry = FakeRegistry(detect_error=ModelUnavailable("missing"))
    with pytest.raises(ModelUnavailable):
        localize_with_yolo_seg(make_settings(tmp_path), registry, image())
