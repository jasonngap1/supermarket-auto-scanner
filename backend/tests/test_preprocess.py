import numpy as np

from app.pipeline.preprocess import PREPROCESS_VERSION, letterbox


def test_letterbox_preserves_aspect_and_pads():
    image = np.full((60, 120, 3), 10, dtype=np.uint8)
    out = letterbox(image, size=224)
    assert out.shape == (224, 224, 3)
    assert tuple(out[0, 0]) == (127, 127, 127)
    assert tuple(out[100, 100]) == (10, 10, 10)


def test_preprocess_version_is_stable():
    assert PREPROCESS_VERSION == "letterbox224-gray10-v1"
