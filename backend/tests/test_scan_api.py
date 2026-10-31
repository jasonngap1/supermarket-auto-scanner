from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient

from app.dependencies import AppState
from app.errors import ModelUnavailable
from app.matching.matcher import GalleryEntry
from app.main import create_app
from app.services.scan import ScanService
from tests.fakes import make_settings


class FakePipeline:
    def __init__(self, error=False):
        self.error = error

    def embed_frame(self, image):
        if self.error:
            raise ModelUnavailable("model unavailable")
        return np.array([1, 0, 0, 0], dtype=np.float32), {"total": 1.0}


class FakeEmbeddings:
    def __init__(self, empty=False):
        self.empty = empty

    def gallery_entries(self):
        if self.empty:
            return []
        return [
            GalleryEntry(
                1, "Milk 1L", 199, np.array([[1, 0, 0, 0]], dtype=np.float32),
                "thumbs/milk.jpg",
            ),
            GalleryEntry(
                2, "Bread", 250, np.array([[0, 1, 0, 0]], dtype=np.float32),
                "thumbs/bread.jpg",
            ),
        ]


def jpeg_bytes():
    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (64, 64), (100, 100, 100)).save(buffer, format="JPEG")
    return buffer.getvalue()


def make_app(tmp_path, *, empty=False, error=False):
    settings = make_settings(tmp_path)
    scan = ScanService(settings, FakePipeline(error=error), FakeEmbeddings(empty=empty))
    return create_app(settings, AppState(settings, scan=scan))


def test_scan_returns_top_items_with_thumbnail(tmp_path):
    with TestClient(make_app(tmp_path)) as client:
        response = client.post(
            "/api/scan",
            files={"image": ("frame.jpg", jpeg_bytes(), "image/jpeg")},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["top_items"][0]["name"] == "Milk 1L"
    assert body["top_items"][0]["price_cents"] == 199
    assert body["top_items"][0]["thumb_rel_path"] == "thumbs/milk.jpg"


def test_scan_reports_no_gallery(tmp_path):
    with TestClient(make_app(tmp_path, empty=True)) as client:
        body = client.post(
            "/api/scan",
            files={"image": ("frame.jpg", jpeg_bytes(), "image/jpeg")},
        ).json()
    assert body["status"] == "no_gallery"
    assert body["top_items"] == []


def test_scan_reports_model_unavailable(tmp_path):
    with TestClient(make_app(tmp_path, error=True)) as client:
        response = client.post(
            "/api/scan",
            files={"image": ("frame.jpg", jpeg_bytes(), "image/jpeg")},
        )
    assert response.status_code == 503
