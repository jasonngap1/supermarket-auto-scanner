import base64
from io import BytesIO
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.dependencies import AppState
from app.matching.matcher import GalleryEntry
from app.main import create_app
from app.pipeline.types import Localization
from app.services.scan import ScanService
from tests.fakes import make_settings


class FakePipeline:
    def __init__(self, status="ok"):
        self.status = status

    def analyze(self, image, variant_ids):
        embeddings = {}
        if self.status == "ok":
            embeddings["v2"] = np.array([1, 0, 0, 0], dtype=np.float32)
        return SimpleNamespace(
            embeddings=embeddings,
            variant_status={"v2": self.status},
            timings_ms={"yolo": 1.0, "embed": 1.0, "total": 2.0},
            localization=Localization(status=self.status, bbox=(5, 5, 40, 40)),
        )


class FakeEmbeddings:
    def __init__(self, empty=False):
        self.empty = empty

    def gallery_entries(self, variant_id="v2"):
        if self.empty:
            return []
        return [
            GalleryEntry(
                1, "Milk 1L", 199,
                {variant_id: np.array([[1, 0, 0, 0]], dtype=np.float32)},
                "thumbs/milk.jpg",
            ),
            GalleryEntry(
                2, "Bread", 250,
                {variant_id: np.array([[0, 1, 0, 0]], dtype=np.float32)},
                "thumbs/bread.jpg",
            ),
        ]


def jpeg_bytes():
    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (64, 64), (100, 100, 100)).save(buffer, format="JPEG")
    return buffer.getvalue()


def make_app(tmp_path, *, empty=False, status="ok"):
    settings = make_settings(tmp_path)
    scan = ScanService(
        settings, FakePipeline(status=status), FakeEmbeddings(empty=empty)
    )
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
    annotated = Image.open(
        BytesIO(base64.b64decode(body["annotated_jpeg_b64"]))
    )
    assert annotated.format == "JPEG"
    assert annotated.size == (64, 64)


def test_scan_reports_no_gallery(tmp_path):
    with TestClient(make_app(tmp_path, empty=True)) as client:
        body = client.post(
            "/api/scan",
            files={"image": ("frame.jpg", jpeg_bytes(), "image/jpeg")},
        ).json()
    assert body["status"] == "no_gallery"
    assert body["top_items"] == []


def test_scan_surfaces_detection_status(tmp_path):
    with TestClient(make_app(tmp_path, status="no_detection")) as client:
        body = client.post(
            "/api/scan",
            files={"image": ("frame.jpg", jpeg_bytes(), "image/jpeg")},
        ).json()
    assert body["status"] == "no_detection"
    assert body["top_items"] == []
    assert body["timings_ms"]["yolo"] == 1.0
