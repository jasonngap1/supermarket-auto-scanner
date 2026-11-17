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
