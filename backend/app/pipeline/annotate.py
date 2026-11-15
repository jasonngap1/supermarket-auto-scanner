from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

from app.domain import TopItem
from app.pipeline.types import Localization


def annotate_frame(
    image_rgb: np.ndarray,
    localization: Localization | None,
    best: TopItem | None,
    currency_symbol: str = "$",
    label: str | None = None,
) -> bytes:
    image = Image.fromarray(image_rgb.copy()).convert("RGB")
    if localization is not None and localization.mask is not None:
        array = np.asarray(image).astype(np.float32)
        mask = localization.mask
        array[mask] = array[mask] * 0.55 + np.array([0, 120, 255]) * 0.45
        image = Image.fromarray(array.clip(0, 255).astype(np.uint8))

    draw = ImageDraw.Draw(image)
    if localization is not None and localization.bbox is not None:
        draw.rectangle(localization.bbox, outline=(0, 200, 255), width=3)

    text = label
    if text is None and best is not None:
        text = f"{best.name}  {currency_symbol}{best.price_cents / 100:.2f}  ({best.score:.2f})"
    if text:
        draw.rectangle((4, 4, 4 + 7 * len(text), 24), fill=(0, 0, 0))
        draw.text((8, 8), text, fill=(255, 255, 255))

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    return buffer.getvalue()
