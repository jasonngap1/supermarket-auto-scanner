from app.errors import ModelUnavailable
from app.pipeline.frame_pipeline import FramePipeline
from app.pipeline.types import Detection
from tests.fakes import FakeRegistry, FakeSam2, image, make_settings


def detection():
    return Detection(0, 46, "banana", 0.9, (20, 20, 80, 80))


def test_v0_alone_never_runs_localization(tmp_path):
    registry = FakeRegistry()
    output = FramePipeline(registry, make_settings(tmp_path)).analyze(image(), ["v0"])
    assert output.variant_status == {"v0": "ok"}
    assert registry.detect_calls == 0
    assert registry.segmenter.calls == 0
    assert output.localization.status == "ok"


def test_v2_uses_yolo_and_sam_once(tmp_path):
    registry = FakeRegistry(detections=[detection()])
    output = FramePipeline(registry, make_settings(tmp_path)).analyze(image(), ["v2"])
    assert output.variant_status == {"v2": "ok"}
    assert output.embeddings["v2"].shape == (8,)
    assert registry.detect_calls == 1
    assert registry.segmenter.calls == 1
    assert output.localization.mask is not None


def test_v0_and_v2_share_one_localization(tmp_path):
    registry = FakeRegistry(detections=[detection()])
    output = FramePipeline(registry, make_settings(tmp_path)).analyze(
        image(), ["v0", "v2"]
    )
    assert output.variant_status == {"v0": "ok", "v2": "ok"}
    assert registry.detect_calls == 1
    assert registry.segmenter.calls == 1


def test_no_detection_is_reported(tmp_path):
    output = FramePipeline(
        FakeRegistry(detections=[]), make_settings(tmp_path)
    ).analyze(image(), ["v2"])
    assert output.variant_status == {"v2": "no_detection"}


def test_mask_rejection_is_reported(tmp_path):
    registry = FakeRegistry(
        segmenter=FakeSam2(empty=True), detections=[detection()]
    )
    output = FramePipeline(registry, make_settings(tmp_path)).analyze(image(), ["v2"])
    assert output.variant_status == {"v2": "mask_rejected"}


def test_model_error_is_reported(tmp_path):
    registry = FakeRegistry(detect_error=ModelUnavailable("missing"))
    output = FramePipeline(registry, make_settings(tmp_path)).analyze(image(), ["v2"])
    assert output.variant_status == {"v2": "model_error"}
