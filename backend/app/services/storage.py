import hashlib
import math
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.config import Settings
from app.errors import ImageTooLarge, InvalidImage


@dataclass(frozen=True)
class CaptureArtifacts:
    rel_path: str
    thumb_rel_path: str
    mask_rel_path: str


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode_image(data: bytes, max_bytes: int) -> tuple[np.ndarray, int, int]:
    if len(data) > max_bytes:
        raise ImageTooLarge(f"Upload exceeds {max_bytes} bytes")
    try:
        image = Image.open(BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImage("Unsupported or corrupt image") from exc
    image = image.convert("RGB")
    return np.asarray(image, dtype=np.uint8), image.width, image.height


def resolve_media_path(settings: Settings, rel_path: str) -> Path:
    candidate = (settings.media_dir / rel_path).resolve()
    media_root = settings.media_dir.resolve()
    if media_root not in candidate.parents:
        raise ValueError("Path escapes media directory")
    return candidate


def save_capture_bundle(
    settings: Settings,
    item_id: int,
    image_rgb: np.ndarray,
    mask: np.ndarray | None = None,
) -> CaptureArtifacts:
    stem = uuid.uuid4().hex
    image_rel = f"images/{item_id}/{stem}.jpg"
    mask_rel = f"masks/{stem}.png" if mask is not None else ""
    thumb_rel = f"thumbs/{stem}.jpg"

    image_path = resolve_media_path(settings, image_rel)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb).save(image_path, format="JPEG", quality=90)

    if mask is not None:
        mask_path = resolve_media_path(settings, mask_rel)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_path, format="PNG")
        overlay = image_rgb.astype(np.float32).copy()
        overlay[mask] = overlay[mask] * 0.55 + np.array([0, 120, 255]) * 0.45
        thumb_image = overlay.clip(0, 255).astype(np.uint8)
    else:
        thumb_image = image_rgb

    thumb = Image.fromarray(thumb_image)
    thumb.thumbnail((settings.thumb_max_px, settings.thumb_max_px))
    thumb_path = resolve_media_path(settings, thumb_rel)
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    thumb.save(thumb_path, format="JPEG", quality=85)

    return CaptureArtifacts(image_rel, thumb_rel, mask_rel)


def load_image_rgb(settings: Settings, rel_path: str) -> np.ndarray:
    with Image.open(resolve_media_path(settings, rel_path)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
