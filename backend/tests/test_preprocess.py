import numpy as np

from app.pipeline.preprocess import (
    CENTER_PREPROCESS_VERSION,
    PREPROCESS_VERSION,
    center_crop,
    letterbox,
    preprocess_for_variant,
)


def test_letterbox_preserves_aspect_and_pads():
    image = np.full((60, 120, 3), 10, dtype=np.uint8)
    out = letterbox(image, size=224)
    assert out.shape == (224, 224, 3)
    assert tuple(out[0, 0]) == (127, 127, 127)
    assert tuple(out[100, 100]) == (10, 10, 10)


def test_preprocess_version_is_stable():
    assert PREPROCESS_VERSION == "letterbox224-gray10-v1"
    assert CENTER_PREPROCESS_VERSION == "center-roi224-gray10-v1"


def test_center_crop_keeps_middle_region():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[15:85, 40:160] = 200
    cropped = center_crop(image)
    assert cropped.shape == (70, 120, 3)
    assert np.all(cropped == 200)
    assert preprocess_for_variant(image, "center").shape == (224, 224, 3)
