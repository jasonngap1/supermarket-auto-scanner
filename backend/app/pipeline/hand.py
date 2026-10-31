import hashlib
from pathlib import Path

import numpy as np

from app.errors import ModelUnavailable
from app.pipeline.types import HandObservation

EXPECTED_HAND_SHA256 = (
    "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
)


def observation_from_landmarks(landmarks, width: int, height: int, score: float = 1.0):
    points = [(float(lm.x) * width, float(lm.y) * height) for lm in landmarks]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return HandObservation(
        landmarks_px=points,
        bbox=(min(xs), min(ys), max(xs), max(ys)),
        score=float(score),
    )


class HandDetector:
    def __init__(self, asset_path: Path, num_hands: int = 1, min_confidence: float = 0.5):
        if not Path(asset_path).exists():
            raise ModelUnavailable(f"Hand landmarker asset not found: {asset_path}")
        digest = hashlib.sha256(Path(asset_path).read_bytes()).hexdigest()
        if digest != EXPECTED_HAND_SHA256:
            raise ModelUnavailable(
                f"Hand landmarker asset checksum mismatch at {asset_path}"
            )
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(asset_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=num_hands,
                min_hand_detection_confidence=min_confidence,
                min_hand_presence_confidence=min_confidence,
                min_tracking_confidence=min_confidence,
            )
        )

    def detect(self, image_rgb: np.ndarray) -> HandObservation | None:
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(image_rgb),
        )
        result = self._landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return None
        score = 1.0
        if result.handedness and result.handedness[0]:
            score = float(result.handedness[0][0].score)
        height, width = image_rgb.shape[:2]
        return observation_from_landmarks(result.hand_landmarks[0], width, height, score)
