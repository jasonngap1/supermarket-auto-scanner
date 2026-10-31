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
            return [MaskCandidate(0, np.zeros((height, width), dtype=bool), 0.1)]
        # Simulate a segmenter that covers the prompted positive point.
        cx, cy = prompt.positive_points[0]
        half = 15
        x0, x1 = max(0, int(cx) - half), min(width, int(cx) + half)
        y0, y1 = max(0, int(cy) - half), min(height, int(cy) + half)
        mask = np.zeros((height, width), dtype=bool)
        mask[y0:y1, x0:x1] = True
        return [
            MaskCandidate(0, mask, 0.9),
            MaskCandidate(1, mask.copy(), 0.7),
            MaskCandidate(2, mask.copy(), 0.5),
        ]


class FakeFace:
    def __init__(self, asset_path=None):
        self.asset_path = asset_path

    def detect(self, image):
        return None


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
    def __init__(self, segmenter=None, embedder=None, face=None):
        self.segmenter = segmenter or FakeSam2()
        self.embedder = embedder or FakeEmbedder()
        self.face = face
        self._gate = threading.RLock()

    def gate(self):
        return self._gate

    def detect_face(self, image):
        return self.face

    def segment(self, image, prompt):
        return self.segmenter.segment(image, prompt)

    def embed(self, image):
        return self.embedder.embed(image)


def image(shape=(100, 100, 3), value=128):
    return np.full(shape, value, dtype=np.uint8)
