import math

import numpy as np
from PIL import Image

from app.pipeline.mask_select import mask_bbox

PREPROCESS_VERSION = "letterbox224-gray10-v1"


def crop_mask(
    image: np.ndarray, mask: np.ndarray, margin_ratio: float = 0.10, background: int = 127
) -> np.ndarray:
    box = mask_bbox(mask)
    if box is None:
        raise ValueError("Empty mask")
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    margin_x = (x2 - x1) * margin_ratio
    margin_y = (y2 - y1) * margin_ratio
    x1 = max(0, int(math.floor(x1 - margin_x)))
    y1 = max(0, int(math.floor(y1 - margin_y)))
    x2 = min(width, int(math.ceil(x2 + margin_x)))
    y2 = min(height, int(math.ceil(y2 + margin_y)))
    crop = image[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]
    out = np.full_like(crop, background)
    out[mask_crop] = crop[mask_crop]
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
