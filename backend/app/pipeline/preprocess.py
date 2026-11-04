import numpy as np
from PIL import Image

from app.pipeline.prompts import center_roi_box

EMBED_VARIANT = "v0"
CENTER_EMBED_VARIANT = "center"
EMBED_VARIANTS = (EMBED_VARIANT, CENTER_EMBED_VARIANT)

PREPROCESS_VERSION = "letterbox224-gray10-v1"
CENTER_PREPROCESS_VERSION = "center-roi224-gray10-v1"
PREPROCESS_VERSIONS = {
    EMBED_VARIANT: PREPROCESS_VERSION,
    CENTER_EMBED_VARIANT: CENTER_PREPROCESS_VERSION,
}


def letterbox(image: np.ndarray, size: int = 224, fill: int = 127) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))
    resized = Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (fill, fill, fill))
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def center_crop(image: np.ndarray, width_ratio: float = 0.60, height_ratio: float = 0.70):
    """Crop the fixed centre region where the held item is expected."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = center_roi_box(width, height, width_ratio, height_ratio)
    left, top = int(round(x1)), int(round(y1))
    right, bottom = int(round(x2)), int(round(y2))
    return image[top:bottom, left:right]


def preprocess_for_variant(image: np.ndarray, variant: str, size: int = 224) -> np.ndarray:
    if variant == CENTER_EMBED_VARIANT:
        return letterbox(center_crop(image), size)
    if variant == EMBED_VARIANT:
        return letterbox(image, size)
    raise ValueError(f"Unknown embedding variant: {variant}")


def preprocess_version_for_variant(variant: str) -> str:
    try:
        return PREPROCESS_VERSIONS[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown embedding variant: {variant}") from exc
