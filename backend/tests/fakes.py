import threading

import numpy as np

from app.config import Settings
from app.pipeline.types import MaskCandidate


def make_settings(tmp_path):
    return Settings(data_dir=tmp_path, device="cpu", warm_models=False)


class FakeSam2:
    def __init__(self, model_id=None, device="cpu", empty=False):
        self.empty = empty
        self.calls = 0
        self.prompt = None

    def segment(self, image, prompt):
        self.calls += 1
        self.prompt = prompt
        height, width = image.shape[:2]
        if self.empty:
            return []
        if prompt.box is not None:
            x1, y1, x2, y2 = prompt.box
        else:
            cx, cy = width / 2.0, height / 2.0
            x1, y1, x2, y2 = cx - 10, cy - 10, cx + 10, cy + 10
        mask = np.zeros((height, width), dtype=bool)
        mask[
            max(0, int(y1)) : min(height, int(y2)),
            max(0, int(x1)) : min(width, int(x2)),
        ] = True
        return [
            MaskCandidate(0, mask, 0.9),
            MaskCandidate(1, mask.copy(), 0.7),
            MaskCandidate(2, mask.copy(), 0.5),
        ]


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
        segmenter=None,
        embedder=None,
        detections=None,
        detect_error=None,
    ):
        self.segmenter = segmenter or FakeSam2()
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

    def segment(self, image, prompt):
        return self.segmenter.segment(image, prompt)

    def embed(self, image):
        return self.embedder.embed(image)


def image(shape=(100, 100, 3), value=128):
    return np.full(shape, value, dtype=np.uint8)
