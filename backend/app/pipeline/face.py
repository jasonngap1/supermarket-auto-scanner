import hashlib
from pathlib import Path

import numpy as np

from app.errors import ModelUnavailable
from app.pipeline.types import FaceObservation

EXPECTED_FACE_SHA256 = (
    "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f"
)


class FaceDetector:
    """MediaPipe face detector used to keep faces out of centre masks."""

    def __init__(
        self,
        asset_path: Path,
        min_confidence: float = 0.5,
    ):
        path = Path(asset_path)
        if not path.exists():
            raise ModelUnavailable(f"Face detector asset not found: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != EXPECTED_FACE_SHA256:
            raise ModelUnavailable(f"Face detector checksum mismatch at {path}")

        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._detector = vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(path)),
                running_mode=vision.RunningMode.IMAGE,
                min_detection_confidence=min_confidence,
            )
        )

    def detect(self, image_rgb: np.ndarray) -> FaceObservation | None:
        height, width = image_rgb.shape[:2]
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(image_rgb),
        )
        result = self._detector.detect(mp_image)
        if not result.detections:
            return None

        detection = max(
            result.detections,
            key=lambda item: item.categories[0].score if item.categories else 0.0,
        )
        box = detection.bounding_box
        x1 = float(min(max(box.origin_x, 0), width - 1))
        y1 = float(min(max(box.origin_y, 0), height - 1))
        x2 = float(min(max(box.origin_x + box.width, 0), width - 1))
        y2 = float(min(max(box.origin_y + box.height, 0), height - 1))
        score = (
            float(detection.categories[0].score) if detection.categories else 0.0
        )
        return FaceObservation(bbox=(x1, y1, x2, y2), score=score)
