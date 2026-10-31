import os

import numpy as np
import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.getenv("RUN_MODEL_TESTS") != "1",
        reason="set RUN_MODEL_TESTS=1 with downloaded model assets",
    ),
]


def test_dinov2_embedding_contract():
    from app.config import Settings
    from app.ml.devices import resolve_device
    from app.pipeline.embedders import DinoV2Embedder

    settings = Settings()
    embedder = DinoV2Embedder(settings.dinov2_model_id, resolve_device(settings.device))
    vector = embedder.embed(np.full((224, 224, 3), 127, dtype=np.uint8))
    assert vector.shape == (384,)
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-3)


def test_sam2_returns_three_deterministic_masks():
    from app.config import Settings
    from app.ml.devices import resolve_device
    from app.pipeline.sam2 import Sam2Segmenter
    from app.pipeline.types import PromptSpec

    settings = Settings()
    sam2 = Sam2Segmenter(settings.sam2_model_id, resolve_device(settings.device))
    image = np.full((480, 640, 3), 127, dtype=np.uint8)
    prompt = PromptSpec([(320.0, 240.0)], [], (200.0, 150.0, 440.0, 330.0), "pinch_roi")
    first = sam2.segment(image, prompt)
    second = sam2.segment(image, prompt)
    assert len(first) == len(second) == 3
    for a, b in zip(first, second):
        union = np.logical_or(a.mask, b.mask).sum()
        intersection = np.logical_and(a.mask, b.mask).sum()
        iou = intersection / union if union else 1.0
        assert iou >= 0.9


def test_sam3_litetext_segment_contract():
    if os.getenv("RUN_SAM3_TESTS") != "1":
        pytest.skip("set RUN_SAM3_TESTS=1 to load the SAM3-LiteText weights")
    from app.config import Settings
    from app.ml.devices import resolve_device
    from app.pipeline.prompts import build_sam3_prompt
    from app.pipeline.sam3 import Sam3LiteSegmenter

    settings = Settings()
    segmenter = Sam3LiteSegmenter(
        settings.sam3_model_id,
        resolve_device(settings.device),
        min_score=settings.sam3_min_score,
        max_candidates=4,
    )
    image = np.full((480, 640, 3), 127, dtype=np.uint8)
    prompt = build_sam3_prompt(640, 480, "object")
    masks = segmenter.segment(image, prompt)
    assert isinstance(masks, list)
    assert len(masks) <= 4
    for mask in masks:
        assert mask.mask.shape == (480, 640)
        assert mask.mask.dtype == bool
        assert 0.0 <= mask.pred_iou <= 1.0


def test_face_asset_loads_and_handles_blank_frame():
    from app.config import Settings
    from app.pipeline.face import FaceDetector

    settings = Settings()
    if not settings.face_asset.exists():
        pytest.skip("blaze_face_short_range.tflite not downloaded")
    detector = FaceDetector(settings.face_asset)
    assert detector.detect(np.zeros((480, 640, 3), dtype=np.uint8)) is None


def test_hand_asset_loads_and_handles_blank_frame():
    from app.config import Settings
    from app.pipeline.hand import HandDetector

    settings = Settings()
    if not settings.hand_asset.exists():
        pytest.skip("hand_landmarker.task not downloaded")
    detector = HandDetector(settings.hand_asset)
    assert detector.detect(np.zeros((480, 640, 3), dtype=np.uint8)) is None


def test_clip_embedding_contract_when_enabled():
    if os.getenv("AUTO_SCANNER_ENABLE_CLIP") != "1":
        pytest.skip("CLIP disabled")
    from app.config import Settings
    from app.ml.devices import resolve_device
    from app.pipeline.embedders import ClipEmbedder

    settings = Settings(enable_clip=True)
    embedder = ClipEmbedder(settings.clip_model_id, resolve_device(settings.device))
    vector = embedder.embed(np.full((224, 224, 3), 127, dtype=np.uint8))
    assert vector.shape == (512,)
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-3)
