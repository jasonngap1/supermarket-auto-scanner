from app.errors import ModelUnavailable
from app.pipeline.frame_pipeline import FramePipeline
from tests.fakes import FakeRegistry, detection, image, make_settings


class BrokenSegmenter:
    def segment(self, image, prompt):
        raise ModelUnavailable("sam2 missing")


def test_v0_alone_never_runs_localization(tmp_path):
    registry = FakeRegistry()
    output = FramePipeline(registry, make_settings(tmp_path)).analyze(image(), ["v0"])
    assert output.variant_status == {"v0": "ok"}
    assert registry.detect_calls == 0
    assert registry.segmenter.calls == 0
    assert output.localization.status == "ok"


def test_v2_uses_yolo_mask_without_sam(tmp_path):
    registry = FakeRegistry(detections=[detection()])
    output = FramePipeline(registry, make_settings(tmp_path)).analyze(image(), ["v2"])
    assert output.variant_status == {"v2": "ok"}
    assert output.embeddings["v2"].shape == (8,)
    assert registry.detect_calls == 1
    assert registry.segmenter.calls == 0
    assert output.localization.mask is not None
    assert set(output.variant_timings["v2"]) == {"yolo"}


def test_v0_and_v2_share_one_yolo_call(tmp_path):
    registry = FakeRegistry(detections=[detection()])
    output = FramePipeline(registry, make_settings(tmp_path)).analyze(
        image(), ["v0", "v2"]
    )
    assert output.variant_status == {"v0": "ok", "v2": "ok"}
    assert registry.detect_calls == 1
    assert registry.segmenter.calls == 0


def test_v2_and_v3_share_one_yolo_call_and_run_sam_once(tmp_path):
    registry = FakeRegistry(detections=[detection()])
    output = FramePipeline(registry, make_settings(tmp_path)).analyze(
        image(), ["v2", "v3"]
    )
    assert output.variant_status == {"v2": "ok", "v3": "ok"}
    assert registry.detect_calls == 1
    assert registry.segmenter.calls == 1
    assert set(output.variant_timings["v2"]) == {"yolo"}
    assert set(output.variant_timings["v3"]) == {"yolo", "sam2"}


def test_no_detection_is_reported(tmp_path):
    output = FramePipeline(
        FakeRegistry(detections=[]), make_settings(tmp_path)
    ).analyze(image(), ["v2"])
    assert output.variant_status == {"v2": "no_detection"}


def test_mask_rejection_is_reported(tmp_path):
    registry = FakeRegistry(detections=[detection(with_mask=False)])
    output = FramePipeline(registry, make_settings(tmp_path)).analyze(image(), ["v2"])
    assert output.variant_status == {"v2": "mask_rejected"}


def test_sam_failure_only_marks_v3(tmp_path):
    registry = FakeRegistry(
        segmenter=BrokenSegmenter(), detections=[detection()]
    )
    output = FramePipeline(registry, make_settings(tmp_path)).analyze(
        image(), ["v2", "v3"]
    )
    assert output.variant_status == {"v2": "ok", "v3": "model_error"}


def test_model_error_is_reported(tmp_path):
    registry = FakeRegistry(detect_error=ModelUnavailable("missing"))
    output = FramePipeline(registry, make_settings(tmp_path)).analyze(image(), ["v2"])
    assert output.variant_status == {"v2": "model_error"}
