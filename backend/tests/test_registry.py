import numpy as np

from app.config import Settings
from app.ml.devices import resolve_device
from app.ml.registry import ModelRegistry


class FakeEmbedder:
    model_key = "dinov2-small"
    dim = 4

    def __init__(self, model_id=None, device="cpu"):
        self.device = device

    def embed(self, image):
        return np.ones(4, dtype=np.float32)


class FakeFace:
    def __init__(self, asset_path=None):
        self.asset_path = asset_path

    def detect(self, image):
        return None


class FakeSegmenter:
    def __init__(self, model_id=None, device="cpu", **kwargs):
        self.model_id = model_id
        self.device = device
        self.prompt = None

    def segment(self, image, prompt):
        self.prompt = prompt
        return []


def make_registry(tmp_path, device="cpu"):
    settings = Settings(data_dir=tmp_path, device=device, warm_models=False)
    return ModelRegistry(
        settings,
        face_factory=FakeFace,
        dinov2_factory=FakeEmbedder,
        sam2_factory=FakeSegmenter,
    )


def test_lazy_loading_and_status(tmp_path):
    registry = make_registry(tmp_path)
    assert registry.status()["models"]["dinov2"]["loaded"] is False
    vector = registry.embed(np.zeros((8, 8, 3), dtype=np.uint8))
    assert vector.shape == (4,)
    assert registry.status()["models"]["dinov2"]["loaded"] is True
    assert registry.status()["device"]["resolved"] == "cpu"


def test_sam2_dispatch_loads_the_segmenter(tmp_path):
    from app.pipeline.prompts import build_center_prompt

    registry = make_registry(tmp_path)
    prompt = build_center_prompt(8, 8)
    assert registry.segment(np.zeros((8, 8, 3), dtype=np.uint8), prompt) == []
    status = registry.status()
    assert status["models"]["sam2"]["loaded"] is True
    assert status["models"]["face"]["loaded"] is False


def test_mps_failure_falls_back_to_cpu(tmp_path):
    settings = Settings(data_dir=tmp_path, device="mps", warm_models=False)
    devices = []

    class FlakyEmbedder(FakeEmbedder):
        def __init__(self, model_id=None, device="cpu"):
            super().__init__(model_id, device)
            devices.append(device)
            self.calls = 0

        def embed(self, image):
            self.calls += 1
            if self.device == "mps" and self.calls == 1:
                raise RuntimeError("MPS kernel failed")
            return np.ones(4, dtype=np.float32)

    registry = ModelRegistry(
        settings,
        face_factory=FakeFace,
        dinov2_factory=FlakyEmbedder,
        sam2_factory=FakeSegmenter,
    )
    registry._device = "mps"
    vector = registry.embed(np.zeros((8, 8, 3), dtype=np.uint8))
    assert vector.shape == (4,)
    assert devices == ["mps", "cpu"]
    assert registry.status()["models"]["dinov2"]["device"] == "cpu"
    assert "MPS kernel failed" in registry.status()["models"]["dinov2"]["error"]


def test_resolve_device():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"mps", "cpu"}
