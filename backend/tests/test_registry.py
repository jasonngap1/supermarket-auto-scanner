import numpy as np
import pytest

from app.config import Settings
from app.errors import ModelUnavailable
from app.ml.devices import resolve_device
from app.ml.registry import ModelRegistry


class FakeEmbedder:
    model_key = "dinov2-small"
    dim = 4

    def __init__(self, model_id=None, device="cpu"):
        self.device = device

    def embed(self, image):
        return np.ones(4, dtype=np.float32)


class FakeHand:
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


def make_registry(tmp_path, device="cpu", localizer="sam3"):
    settings = Settings(
        data_dir=tmp_path, device=device, warm_models=False, localizer=localizer
    )
    return ModelRegistry(
        settings,
        hand_factory=FakeHand,
        dinov2_factory=FakeEmbedder,
        clip_factory=FakeEmbedder,
        sam2_factory=lambda model_id, device: FakeEmbedder(model_id, device),
        sam3_factory=FakeSegmenter,
    )


def test_lazy_loading_and_status(tmp_path):
    registry = make_registry(tmp_path)
    assert registry.status()["models"]["dinov2"]["loaded"] is False
    vector = registry.embed("dinov2", np.zeros((8, 8, 3), dtype=np.uint8))
    assert vector.shape == (4,)
    assert registry.status()["models"]["dinov2"]["loaded"] is True
    assert registry.status()["device"]["resolved"] == "cpu"


def test_sam3_dispatch_loads_the_selected_localizer(tmp_path):
    from app.pipeline.types import PromptSpec

    registry = make_registry(tmp_path)
    prompt = PromptSpec([], [], None, "sam3_center", text="product")
    assert registry.segment(np.zeros((8, 8, 3), dtype=np.uint8), prompt) == []
    status = registry.status()
    assert status["localizer"] == "sam3"
    assert status["models"]["sam3"]["loaded"] is True
    assert status["models"]["sam2"]["loaded"] is False


def test_sam2_center_dispatch_uses_the_sam2_model(tmp_path):
    from app.pipeline.types import PromptSpec

    settings = Settings(
        data_dir=tmp_path, device="cpu", warm_models=False, localizer="sam2_center"
    )
    registry = ModelRegistry(
        settings,
        hand_factory=FakeHand,
        dinov2_factory=FakeEmbedder,
        clip_factory=FakeEmbedder,
        sam2_factory=FakeSegmenter,
        sam3_factory=FakeSegmenter,
    )
    prompt = PromptSpec([(4.0, 4.0)], [], (0.0, 0.0, 8.0, 8.0), "sam2_center")
    assert registry.segment(np.zeros((8, 8, 3), dtype=np.uint8), prompt) == []
    status = registry.status()
    assert status["localizer"] == "sam2_center"
    assert status["models"]["sam2"]["loaded"] is True
    assert status["models"]["sam3"]["loaded"] is False


def test_clip_is_unavailable_unless_enabled(tmp_path):
    registry = make_registry(tmp_path)
    with pytest.raises(ModelUnavailable):
        registry.embed("clip", np.zeros((8, 8, 3), dtype=np.uint8))


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
        hand_factory=FakeHand,
        dinov2_factory=FlakyEmbedder,
        clip_factory=FakeEmbedder,
        sam2_factory=lambda model_id, device: FlakyEmbedder(model_id, device),
    )
    registry._device = "mps"
    vector = registry.embed("dinov2", np.zeros((8, 8, 3), dtype=np.uint8))
    assert vector.shape == (4,)
    assert devices == ["mps", "cpu"]
    assert registry.status()["models"]["dinov2"]["device"] == "cpu"
    assert "MPS kernel failed" in registry.status()["models"]["dinov2"]["error"]


def test_resolve_device():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"mps", "cpu"}
