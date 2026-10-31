import numpy as np

from app.pipeline.preprocess import PREPROCESS_VERSION, crop_box, crop_mask, letterbox


def test_letterbox_preserves_aspect_and_pads():
    image = np.full((60, 120, 3), 10, dtype=np.uint8)
    out = letterbox(image, size=224)
    assert out.shape == (224, 224, 3)
    assert tuple(out[0, 0]) == (127, 127, 127)
    assert tuple(out[100, 100]) == (10, 10, 10)


def test_crop_mask_grays_outside_mask():
    image = np.full((40, 40, 3), 200, dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    mask[15:25, 15:25] = False
    out = crop_mask(image, mask, margin_ratio=0.0)
    assert out.shape == (20, 20, 3)
    assert tuple(out[0, 0]) == (200, 200, 200)
    assert tuple(out[10, 10]) == (127, 127, 127)


def test_crop_box_clips_and_version():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    crop = crop_box(image, (-5, -5, 10, 10), margin_ratio=0.0)
    assert crop.shape[:2] == (10, 10)
    assert PREPROCESS_VERSION == "letterbox224-gray10-v1"
