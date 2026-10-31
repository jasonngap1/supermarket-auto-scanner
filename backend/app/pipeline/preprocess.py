import numpy as np
from PIL import Image

PREPROCESS_VERSION = "letterbox224-gray10-v1"


def letterbox(image: np.ndarray, size: int = 224, fill: int = 127) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))
    resized = Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (fill, fill, fill))
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return np.asarray(canvas, dtype=np.uint8)
