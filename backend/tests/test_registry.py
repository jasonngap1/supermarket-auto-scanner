import numpy as np
import pytest

from app.config import Settings
from app.errors import ModelUnavailable
from app.ml.devices import resolve_device
from app.ml.registry import ModelRegistry
from app.pipeline.types import Detection


class FakeEmbedder:
    dim = 4

    def __init__(self, model_id=None, device="cpu"):
        self.model_id = model_id
        self.device = device

    def embed(self, image):
        return np.ones(4, dtype=np.float32)


class FakeYolo:
    def __init__(self, model_id, device, **kwargs):
        self.model_id = model_id
        self.device = device
        self.kwargs = kwargs

    def predict(self, image):
        return [Detection(0, 46, "banana", 0.9, (1, 1, 5, 5))]


def make_registry(tmp_path):
    settings = Settings(data_dir=tmp_path, device="cpu", warm_models=False)
    return ModelRegistry(
        settings,
        dinov2_factory=FakeEmbedder,
        yolo_factory=FakeYolo,
    )


def test_lazy_loading_and_status(tmp_path):
    registry = make_registry(tmp_path)
    status = registry.status()["models"]
    assert status["dinov2"]["loaded"] is False
    assert status["yolo"]["loaded"] is False

    vector = registry.embed(np.zeros((8, 8, 3), dtype=np.uint8))
    assert vector.shape == (4,)
    assert registry.status()["models"]["dinov2"]["loaded"] is True
    assert registry.status()["device"]["resolved"] == resolve_device("cpu")


def test_detect_objects_dispatch(tmp_path):
    registry = make_registry(tmp_path)
    detections = registry.detect_objects(np.zeros((8, 8, 3), dtype=np.uint8))
    assert detections[0].class_name == "banana"


def test_model_errors_are_wrapped(tmp_path):
    class Broken:
        def __init__(self, **kwargs):
            raise RuntimeError("boom")

    settings = Settings(data_dir=tmp_path, device="cpu", warm_models=False)
    registry = ModelRegistry(settings, dinov2_factory=Broken)
    with pytest.raises(ModelUnavailable):
        registry.embed(np.zeros((8, 8, 3), dtype=np.uint8))
