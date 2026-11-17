import math

import numpy as np
from PIL import Image

from app.pipeline.mask_select import mask_bbox

PREPROCESS_VERSION = "letterbox224-gray10-v1"


def _mask_crop_bounds(
    image: np.ndarray, mask: np.ndarray, margin_ratio: float
) -> tuple[np.ndarray, np.ndarray] | None:
    box = mask_bbox(mask)
    if box is None:
        return None
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    margin_x = (x2 - x1) * margin_ratio
    margin_y = (y2 - y1) * margin_ratio
    x1 = max(0, int(math.floor(x1 - margin_x)))
    y1 = max(0, int(math.floor(y1 - margin_y)))
    x2 = min(width, int(math.ceil(x2 + margin_x)))
    y2 = min(height, int(math.ceil(y2 + margin_y)))
    return image[y1:y2, x1:x2], mask[y1:y2, x1:x2]


def normalize_mask_crop(
    image: np.ndarray,
    mask: np.ndarray,
    margin_ratio: float = 0.10,
    background: int = 127,
    *,
    target_mean: float = 128.0,
    target_std: float = 50.0,
    min_pixels: int = 16,
) -> np.ndarray:
    """Canonical photometric space for masked crops.

    Gray-world white balance removes a warm or cool light cast, then per-channel
    contrast standardization brings gallery and realtime crops to the same
    brightness range. Statistics use masked pixels only, so the gray background
    never influences the result. The same function runs at enrollment and scan
    time, which is what keeps the vectors comparable.
    """
    cropped = _mask_crop_bounds(image, mask, margin_ratio)
    if cropped is None:
        raise ValueError("Empty mask")
    crop, mask_crop = cropped
    if int(mask_crop.sum()) < min_pixels:
        crop_values = crop.astype(np.float32)
    else:
        values = crop[mask_crop].astype(np.float32)
        means = values.mean(axis=0)
        target = float(means.mean())
        gains = np.ones(3, dtype=np.float32)
        for channel in range(3):
            if means[channel] > 1.0:
                gains[channel] = target / means[channel]
        gains = np.clip(gains, 0.5, 2.0)
        balanced = crop.astype(np.float32) * gains

        balanced_values = balanced[mask_crop]
        channel_means = balanced_values.mean(axis=0)
        channel_stds = balanced_values.std(axis=0)
        if float(channel_stds.max()) < 1.0:
            # Flat or nearly flat crop: keep its brightness rather than
            # amplifying sensor noise into a false texture.
            crop_values = balanced
        else:
            scales = np.zeros(3, dtype=np.float32)
            for channel in range(3):
                if channel_stds[channel] > 1.0:
                    scales[channel] = target_std / channel_stds[channel]
            crop_values = (
                (balanced - channel_means) * scales + target_mean
            )
    crop_values = np.clip(crop_values, 0, 255)
    out = np.full_like(crop, background)
    out[mask_crop] = crop_values[mask_crop].astype(np.uint8)
    return out


def letterbox(image: np.ndarray, size: int = 224, fill: int = 127) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))
    resized = Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (fill, fill, fill))
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return np.asarray(canvas, dtype=np.uint8)
