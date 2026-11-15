import pytest

from app.errors import ModelUnavailable
from app.pipeline.types import Detection
from app.pipeline.yolo import localize_with_yolo_sam, parse_class_ids
from tests.fakes import FakeRegistry, FakeSam2, image, make_settings


def test_parse_class_ids_accepts_names_and_ids():
    names = {0: "person", 46: "banana"}
    assert parse_class_ids("banana,0", names) == [46, 0]
    assert parse_class_ids("", names) is None


def test_parse_class_ids_rejects_unknown_names():
    with pytest.raises(ModelUnavailable):
        parse_class_ids("unicorn", {0: "person"})


def test_detection_box_is_segmented_and_selected(tmp_path):
    detection = Detection(0, 46, "banana", 0.9, (20, 20, 80, 80))
    registry = FakeRegistry(detections=[detection])
    localization = localize_with_yolo_sam(make_settings(tmp_path), registry, image())
    assert localization.status == "ok"
    assert localization.bbox == (20.0, 20.0, 80.0, 80.0)
    assert localization.mask is not None and localization.mask.sum() == 3600
    assert registry.segmenter.prompt.box == (20.0, 20.0, 80.0, 80.0)
    assert registry.segmenter.prompt.positive_points == [(50.0, 50.0)]
    assert localization.timings_ms.keys() == {"yolo", "sam2"}


def test_no_detection_reports_no_detection(tmp_path):
    localization = localize_with_yolo_sam(
        make_settings(tmp_path), FakeRegistry(detections=[]), image()
    )
    assert localization.status == "no_detection"
    assert localization.mask is None


def test_rejected_class_reports_no_detection(tmp_path):
    detection = Detection(0, 0, "person", 0.9, (20, 20, 80, 80))
    localization = localize_with_yolo_sam(
        make_settings(tmp_path), FakeRegistry(detections=[detection]), image()
    )
    assert localization.status == "no_detection"
    assert localization.detection_selection.reasons == ["rejected_class"]


def test_empty_sam_result_reports_mask_rejected(tmp_path):
    detection = Detection(0, 46, "banana", 0.9, (20, 20, 80, 80))
    registry = FakeRegistry(segmenter=FakeSam2(empty=True), detections=[detection])
    localization = localize_with_yolo_sam(make_settings(tmp_path), registry, image())
    assert localization.status == "mask_rejected"
    assert localization.bbox == (20.0, 20.0, 80.0, 80.0)
    assert localization.mask is None


def test_detector_failure_propagates(tmp_path):
    registry = FakeRegistry(detect_error=ModelUnavailable("missing"))
    with pytest.raises(ModelUnavailable):
        localize_with_yolo_sam(make_settings(tmp_path), registry, image())
