from app.pipeline.frame_pipeline import FramePipeline
from app.pipeline.types import FaceObservation
from tests.fakes import FakeRegistry, FakeSam2, image, make_settings


def test_v0_embedding_never_runs_segmentation(tmp_path):
    segmenter = FakeSam2()
    pipeline = FramePipeline(FakeRegistry(segmenter=segmenter), make_settings(tmp_path))
    vector, timings = pipeline.embed_frame(image())
    assert vector.shape == (8,)
    assert segmenter.calls == 0
    assert timings["preprocess_v0"] >= 0
    assert timings["embed_v0"] >= 0


def test_embedding_variants_include_whole_frame_and_center_crop(tmp_path):
    segmenter = FakeSam2()
    pipeline = FramePipeline(FakeRegistry(segmenter=segmenter), make_settings(tmp_path))
    vectors, timings = pipeline.embed_frame_variants(image())
    assert set(vectors) == {"v0", "center"}
    assert all(vector.shape == (8,) for vector in vectors.values())
    assert segmenter.calls == 0
    assert set(timings) == {
        "preprocess_v0", "embed_v0", "preprocess_center", "embed_center", "total"
    }


def test_sam2_center_uses_centre_point_without_a_hand(tmp_path):
    segmenter = FakeSam2()
    pipeline = FramePipeline(
        FakeRegistry(segmenter=segmenter), make_settings(tmp_path)
    )
    localization = pipeline.localize(image())
    assert localization.status == "ok"
    assert localization.mask is not None
    assert segmenter.prompt.positive_points == [(50.0, 50.0)]
    assert segmenter.prompt.box == (20.0, 15.0, 80.0, 85.0)
    assert len(segmenter.prompt.negative_points) == 12


def test_sam2_center_rejects_a_face_mask(tmp_path):
    face = FaceObservation(bbox=(35.0, 35.0, 65.0, 65.0), score=0.9)
    segmenter = FakeSam2()
    pipeline = FramePipeline(
        FakeRegistry(segmenter=segmenter, face=face), make_settings(tmp_path)
    )
    localization = pipeline.localize(image())
    assert localization.status == "mask_rejected"
    assert localization.mask is None
    assert localization.face is not None
    assert len(segmenter.prompt.negative_points) == 12


def test_sam2_center_uses_visible_face_as_negative_prompts(tmp_path):
    face = FaceObservation(bbox=(75.0, 5.0, 95.0, 25.0), score=0.9)
    segmenter = FakeSam2()
    pipeline = FramePipeline(
        FakeRegistry(segmenter=segmenter, face=face), make_settings(tmp_path)
    )
    localization = pipeline.localize(image())
    assert localization.status == "ok"
    assert localization.face is not None
    assert len(segmenter.prompt.negative_points) == 17


def test_sam2_center_rejects_a_mask_without_the_centre_point(tmp_path):
    pipeline = FramePipeline(
        FakeRegistry(segmenter=FakeSam2(empty=True)), make_settings(tmp_path)
    )
    localization = pipeline.localize(image())
    assert localization.status == "mask_rejected"
    assert localization.mask is None
