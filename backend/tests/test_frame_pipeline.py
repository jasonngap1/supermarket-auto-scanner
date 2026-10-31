from app.pipeline.frame_pipeline import FramePipeline
from app.pipeline.types import FaceObservation
from tests.fakes import FakeHand, FakeRegistry, FakeSam2, FakeSam3, NoHand, image, make_settings


def test_sam3_uses_hand_for_v1_but_not_the_mask(tmp_path):
    pipeline = FramePipeline(FakeRegistry(), make_settings(tmp_path))
    output = pipeline.analyze(image(), ["v0", "v1", "v2"])
    assert output.variant_status == {"v0": "ok", "v1": "ok", "v2": "ok"}
    assert output.localization.status == "ok"
    assert set(output.embeddings) == {"v0", "v1", "v2"}
    assert output.timings_ms["hand"] >= 0
    assert output.timings_ms["sam3"] >= 0


def test_sam3_mask_works_without_a_hand(tmp_path):
    pipeline = FramePipeline(
        FakeRegistry(hand=NoHand()), make_settings(tmp_path)
    )
    output = pipeline.analyze(image(), ["v0", "v1", "v2"])
    assert output.variant_status == {"v0": "ok", "v1": "no_hand", "v2": "ok"}
    assert output.localization.status == "ok"
    assert output.localization.mask is not None
    assert "v0" in output.embeddings and "v2" in output.embeddings
    assert "v1" not in output.embeddings


def test_sam2_path_still_requires_a_hand(tmp_path):
    pipeline = FramePipeline(
        FakeRegistry(hand=NoHand(), segmenter=FakeSam2()),
        make_settings(tmp_path, localizer="sam2"),
    )
    output = pipeline.analyze(image(), ["v0", "v1", "v2"])
    assert output.variant_status == {"v0": "ok", "v1": "no_hand", "v2": "no_hand"}
    assert output.localization.status == "no_hand"
    assert "v0" in output.embeddings and "v2" not in output.embeddings


def test_sam2_path_runs_prompt_segmentation(tmp_path):
    pipeline = FramePipeline(
        FakeRegistry(segmenter=FakeSam2()),
        make_settings(tmp_path, localizer="sam2"),
    )
    output = pipeline.analyze(image(), ["v0", "v1", "v2"])
    assert output.variant_status == {"v0": "ok", "v1": "ok", "v2": "ok"}
    assert output.localization.status == "ok"
    assert output.timings_ms["sam2"] >= 0


def test_sam2_center_uses_centre_point_without_a_hand(tmp_path):
    segmenter = FakeSam2()
    pipeline = FramePipeline(
        FakeRegistry(hand=NoHand(), segmenter=segmenter),
        make_settings(tmp_path, localizer="sam2_center"),
    )
    output = pipeline.analyze(image(), ["v0", "v1", "v2"])
    assert output.variant_status == {"v0": "ok", "v1": "no_hand", "v2": "ok"}
    assert output.localization.status == "ok"
    assert output.localization.mask is not None
    assert segmenter.prompt.positive_points == [(50.0, 50.0)]
    assert segmenter.prompt.box == (20.0, 15.0, 80.0, 85.0)
    assert output.timings_ms["sam2"] >= 0


def test_sam2_center_rejects_a_face_mask(tmp_path):
    face = FaceObservation(bbox=(35.0, 35.0, 65.0, 65.0), score=0.9)
    segmenter = FakeSam2()
    pipeline = FramePipeline(
        FakeRegistry(hand=NoHand(), segmenter=segmenter, face=face),
        make_settings(tmp_path, localizer="sam2_center"),
    )
    output = pipeline.analyze(image(), ["v0", "v2"])
    assert output.localization.status == "mask_rejected"
    assert output.localization.face is not None
    assert output.variant_status == {"v0": "ok", "v2": "mask_rejected"}
    assert len(segmenter.prompt.negative_points) == 12


def test_sam2_center_uses_visible_face_as_negative_prompts(tmp_path):
    face = FaceObservation(bbox=(75.0, 5.0, 95.0, 25.0), score=0.9)
    segmenter = FakeSam2()
    pipeline = FramePipeline(
        FakeRegistry(hand=NoHand(), segmenter=segmenter, face=face),
        make_settings(tmp_path, localizer="sam2_center"),
    )
    output = pipeline.analyze(image(), ["v0", "v2"])
    assert output.localization.status == "ok"
    assert output.localization.face is not None
    assert len(segmenter.prompt.negative_points) == 17


def test_sam2_center_rejects_a_mask_without_the_centre_point(tmp_path):
    pipeline = FramePipeline(
        FakeRegistry(hand=NoHand(), segmenter=FakeSam2(empty=True)),
        make_settings(tmp_path, localizer="sam2_center"),
    )
    output = pipeline.analyze(image(), ["v1", "v2"])
    assert output.localization.status == "mask_rejected"
    assert output.variant_status == {"v1": "no_hand", "v2": "mask_rejected"}


def test_rejected_mask_leaves_v1_as_the_control(tmp_path):
    pipeline = FramePipeline(
        FakeRegistry(segmenter=FakeSam3(empty=True)), make_settings(tmp_path)
    )
    output = pipeline.analyze(image(), ["v1", "v2"])
    assert output.localization.status == "mask_rejected"
    assert output.variant_status == {"v1": "ok", "v2": "mask_rejected"}


def test_v0_alone_never_runs_localization(tmp_path):
    hand, segmenter = FakeHand(), FakeSam3()
    pipeline = FramePipeline(
        FakeRegistry(hand=hand, segmenter=segmenter), make_settings(tmp_path)
    )
    pipeline.analyze(image(), ["v0"])
    assert hand.calls == 0 and segmenter.calls == 0
