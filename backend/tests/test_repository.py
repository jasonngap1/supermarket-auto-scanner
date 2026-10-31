import numpy as np
import pytest

from app.config import Settings
from app.db import init_db
from app.errors import DuplicateCapture
from app.services.captures import (
    add_capture,
    delete_capture,
    get_embedding,
    list_captures,
    upsert_embedding,
)
from app.services.items import create_item
from app.services.storage import save_capture_bundle


def setup_item(tmp_path):
    settings = Settings(data_dir=tmp_path)
    init_db(settings)
    item = create_item(settings, "Milk", 199)
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    mask = np.ones((32, 32), dtype=bool)
    artifacts = save_capture_bundle(settings, item.id, image, mask)
    return settings, item, artifacts


def make_capture(settings, item, artifacts, role="gallery", sha="a" * 64):
    return add_capture(
        settings,
        item_id=item.id,
        role=role,
        session_id="s1",
        artifacts=artifacts,
        sha256=sha,
        width=32,
        height=32,
        mask_found=True,
        mask_status="ok",
        localization={"status": "ok", "selection": {}},
    )


def test_capture_duplicate_and_cascade_delete(tmp_path):
    settings, item, artifacts = setup_item(tmp_path)
    capture = make_capture(settings, item, artifacts)
    assert list_captures(settings, item.id)[0].id == capture.id
    with pytest.raises(DuplicateCapture):
        make_capture(settings, item, artifacts)
    assert delete_capture(settings, item.id, capture.id) is True
    assert list_captures(settings, item.id) == []


def test_embedding_round_trip(tmp_path):
    settings, item, artifacts = setup_item(tmp_path)
    capture = make_capture(settings, item, artifacts)
    vector = np.linspace(0, 1, 384, dtype=np.float32)
    upsert_embedding(settings, capture.id, "v0", "dinov2-small", "prep-v1", vector)
    loaded = get_embedding(settings, capture.id, "v0", "dinov2-small", "prep-v1")
    assert loaded is not None
    assert np.allclose(loaded, vector)
    assert get_embedding(settings, capture.id, "v1", "dinov2-small", "prep-v1") is None
