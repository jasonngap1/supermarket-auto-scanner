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
    from app.pipeline.prompts import build_center_prompt
    from app.pipeline.sam2 import Sam2Segmenter

    settings = Settings()
    sam2 = Sam2Segmenter(settings.sam2_model_id, resolve_device(settings.device))
    image = np.full((480, 640, 3), 127, dtype=np.uint8)
    prompt = build_center_prompt(640, 480)
    first = sam2.segment(image, prompt)
    second = sam2.segment(image, prompt)
    assert len(first) == len(second) == 3
    for a, b in zip(first, second):
        union = np.logical_or(a.mask, b.mask).sum()
        intersection = np.logical_and(a.mask, b.mask).sum()
        iou = intersection / union if union else 1.0
        assert iou >= 0.9


def test_face_asset_loads_and_handles_blank_frame():
    from app.config import Settings
    from app.pipeline.face import FaceDetector

    settings = Settings()
    if not settings.face_asset.exists():
        pytest.skip("blaze_face_short_range.tflite not downloaded")
    detector = FaceDetector(settings.face_asset)
    assert detector.detect(np.zeros((480, 640, 3), dtype=np.uint8)) is None
