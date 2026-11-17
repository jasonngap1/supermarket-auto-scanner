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
    prompt = PromptSpec([(320.0, 240.0)], (200.0, 150.0, 440.0, 330.0))
    first = sam2.segment(image, prompt)
    second = sam2.segment(image, prompt)
    assert len(first) == len(second) == 3
    for a, b in zip(first, second):
        union = np.logical_or(a.mask, b.mask).sum()
        intersection = np.logical_and(a.mask, b.mask).sum()
        iou = intersection / union if union else 1.0
        assert iou >= 0.9


def test_yolo_contract():
    if os.getenv("RUN_YOLO_TESTS") != "1":
        pytest.skip("set RUN_YOLO_TESTS=1 with downloaded YOLO weights")
    pytest.importorskip("ultralytics")
    pytest.importorskip("onnxruntime")

    from app.config import Settings
    from app.ml.registry import ModelRegistry

    settings = Settings(warm_models=False)
    weights = settings.data_dir / "weights" / settings.yolo_seg_model_id
    if not weights.exists():
        pytest.skip(f"missing {weights}; run scripts/download_models.py")
    registry = ModelRegistry(settings)
    frame = np.full((480, 640, 3), 127, dtype=np.uint8)
    detections = registry.detect_objects(frame)
    assert isinstance(detections, list)
    for detection in detections:
        assert 0.0 <= detection.confidence <= 1.0
        assert len(detection.bbox) == 4
        assert detection.mask is None or detection.mask.dtype == bool
