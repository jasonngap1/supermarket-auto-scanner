import threading

import numpy as np

from app.config import Settings
from app.pipeline.types import FaceObservation, HandObservation, MaskCandidate


def make_settings(tmp_path, localizer="sam3"):
    return Settings(
        data_dir=tmp_path,
        device="cpu",
        warm_models=False,
        enable_clip=True,
        localizer=localizer,
    )


class FakeHand:
    def __init__(self, asset_path=None):
        self.asset_path = asset_path
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        points = [(10.0, 10.0)] * 21
        points[0] = (10.0, 10.0)
        points[4] = (50.0, 30.0)
        points[5] = (12.0, 12.0)
        points[8] = (52.0, 30.0)
        points[9] = (14.0, 14.0)
        points[12] = (55.0, 32.0)
        points[13] = (16.0, 16.0)
        points[17] = (18.0, 18.0)
        return HandObservation(
            landmarks_px=points,
            bbox=(10.0, 10.0, 55.0, 32.0),
            score=0.9,
        )


class NoHand:
    def __init__(self, asset_path=None):
        pass

    def detect(self, image):
        return None


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


class FakeSam3:
    def __init__(self, model_id=None, device="cpu", empty=False, **kwargs):
        self.empty = empty
        self.calls = 0

    def segment(self, image, prompt):
        self.calls += 1
        height, width = image.shape[:2]
        if self.empty:
            return []
        # Segment the centre of the search region.
        if prompt.box is not None:
            x1, y1, x2, y2 = prompt.box
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        else:
            cx, cy = width / 2.0, height / 2.0
        half = max(4, min(width, height) // 10)
        x0, x1 = max(0, int(cx) - half), min(width, int(cx) + half)
        y0, y1 = max(0, int(cy) - half), min(height, int(cy) + half)
        mask = np.zeros((height, width), dtype=bool)
        mask[y0:y1, x0:x1] = True
        return [
            MaskCandidate(0, mask, 0.9),
            MaskCandidate(1, mask.copy(), 0.6),
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
    def __init__(self, hand=None, segmenter=None, embedder=None, face=None):
        self.hand = hand or FakeHand()
        self.segmenter = segmenter or FakeSam3()
        self.embedder = embedder or FakeEmbedder()
        self.face = face
        self._gate = threading.RLock()

    def gate(self):
        return self._gate

    def detect_hand(self, image):
        return self.hand.detect(image)

    def detect_face(self, image):
        return self.face

    def segment(self, image, prompt):
        return self.segmenter.segment(image, prompt)

    def embed(self, kind, image):
        return self.embedder.embed(image)


def image(shape=(100, 100, 3), value=128):
    return np.full(shape, value, dtype=np.uint8)
