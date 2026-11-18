from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from app.config import Settings
from app.errors import ImageTooLarge, InvalidImage
from app.services.storage import (
    decode_image,
    resolve_media_path,
    save_capture_bundle,
    sha256_hex,
)


def jpeg_bytes(width=64, height=48, color=(200, 30, 30)) -> bytes:
    image = Image.new("RGB", (width, height), color)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_decode_rejects_bad_and_large_payloads():
    with pytest.raises(InvalidImage):
        decode_image(b"not an image", max_bytes=1000)
    with pytest.raises(ImageTooLarge):
        decode_image(jpeg_bytes(), max_bytes=10)


def test_save_capture_bundle_and_path_safety(tmp_path):
    settings = Settings(data_dir=tmp_path)
    image, width, height = decode_image(jpeg_bytes(), max_bytes=10_000)
    mask = np.zeros((height, width), dtype=bool)
    mask[10:40, 10:50] = True
    artifacts = save_capture_bundle(settings, item_id=7, image_rgb=image, mask=mask)
    for rel_path in (artifacts.rel_path, artifacts.thumb_rel_path, artifacts.mask_rel_path):
        path = resolve_media_path(settings, rel_path)
        assert path.exists()
        assert settings.media_dir.resolve() in path.parents
    assert (width, height) == (64, 48)
    assert len(sha256_hex(jpeg_bytes())) == 64


def test_save_capture_bundle_without_mask(tmp_path):
    settings = Settings(data_dir=tmp_path)
    image, _width, _height = decode_image(jpeg_bytes(), max_bytes=10_000)
    artifacts = save_capture_bundle(settings, item_id=7, image_rgb=image)
    assert artifacts.mask_rel_path == ""
    assert resolve_media_path(settings, artifacts.rel_path).exists()
    assert resolve_media_path(settings, artifacts.thumb_rel_path).exists()


def test_rejects_path_traversal(tmp_path):
    settings = Settings(data_dir=tmp_path)
    with pytest.raises(ValueError):
        resolve_media_path(settings, "../app.db")
