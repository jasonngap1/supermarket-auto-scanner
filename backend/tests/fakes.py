import threading

import numpy as np

from app.config import Settings
from app.pipeline.types import Detection


def make_settings(tmp_path):
    return Settings(data_dir=tmp_path, device="cpu", warm_models=False)


def box_mask(bbox=(20, 20, 80, 80), shape=(100, 100)):
    mask = np.zeros(shape, dtype=bool)
    x1, y1, x2, y2 = (int(value) for value in bbox)
    mask[y1:y2, x1:x2] = True
    return mask


def detection(
    bbox=(20, 20, 80, 80),
    *,
    class_id=46,
    class_name="banana",
    confidence=0.9,
    with_mask=True,
):
    return Detection(
        index=0,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=tuple(float(value) for value in bbox),
        mask=box_mask(bbox) if with_mask else None,
    )


class FakeEmbedder:
    dim = 8

    def __init__(self, model_id=None, device="cpu", constant=None):
        self.constant = constant

    def embed(self, image):
        if self.constant is not None:
            return np.asarray(self.constant, dtype=np.float32)
        value = float(np.mean(image)) / 255.0
        return np.full(self.dim, value, dtype=np.float32)


class FakeRegistry:
    def __init__(
        self,
        embedder=None,
        detections=None,
        detect_error=None,
    ):
        self.embedder = embedder or FakeEmbedder()
        self.detections = detections or []
        self.detect_error = detect_error
        self.detect_calls = 0
        self._gate = threading.RLock()

    def gate(self):
        return self._gate

    def detect_objects(self, image):
        self.detect_calls += 1
        if self.detect_error is not None:
            raise self.detect_error
        return self.detections

    def embed(self, image):
        return self.embedder.embed(image)


def image(shape=(100, 100, 3), value=128):
    return np.full(shape, value, dtype=np.uint8)
