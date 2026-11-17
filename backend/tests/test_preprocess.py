import numpy as np

from app.pipeline.preprocess import (
    PREPROCESS_VERSION,
    letterbox,
    normalize_mask_crop,
)


def test_letterbox_preserves_aspect_and_pads():
    image = np.full((60, 120, 3), 10, dtype=np.uint8)
    out = letterbox(image, size=224)
    assert out.shape == (224, 224, 3)
    assert tuple(out[0, 0]) == (127, 127, 127)
    assert tuple(out[100, 100]) == (10, 10, 10)


def test_preprocess_version_is_stable():
    assert PREPROCESS_VERSION == "letterbox224-gray10-v1"


def test_normalize_mask_crop_neutralizes_a_color_cast():
    height = width = 64
    ramp = np.linspace(0, 20, width, dtype=np.float32)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = np.clip(180 + ramp, 0, 255)
    image[..., 1] = np.clip(140 + ramp, 0, 255)
    image[..., 2] = np.clip(100 + ramp, 0, 255)
    mask = np.zeros((height, width), dtype=bool)
    mask[8:56, 8:56] = True
    mask[28:36, 28:36] = False

    out = normalize_mask_crop(image, mask, margin_ratio=0.0, target_std=20)

    means = out[mask[8:56, 8:56]].mean(axis=0)
    assert np.all(np.abs(means - 128.0) <= 2.0)
    assert tuple(out[20, 20]) == (127, 127, 127)


def test_normalize_mask_crop_keeps_a_flat_crop_unchanged():
    image = np.full((40, 40, 3), 200, dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    mask[15:25, 15:25] = False

    out = normalize_mask_crop(image, mask, margin_ratio=0.0)

    assert np.all(out[mask[10:30, 10:30]] == 200)
    assert tuple(out[10, 10]) == (127, 127, 127)


def test_normalize_mask_crop_is_deterministic_and_guards_few_pixels():
    image = np.full((40, 40, 3), 90, dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:11, 10:13] = True  # 3 pixels, below the guard

    first = normalize_mask_crop(image, mask, margin_ratio=0.0)
    second = normalize_mask_crop(image, mask, margin_ratio=0.0)

    assert np.array_equal(first, second)
    assert np.all(first[mask[10:11, 10:13]] == 90)
