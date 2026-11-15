from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.dependencies import AppState
from app.matching.matcher import GalleryEntry
from app.main import create_app
from app.pipeline.frame_pipeline import PipelineOutput
from app.pipeline.types import Localization
from app.services.analysis import AnalysisService
from tests.fakes import make_settings


class FakePipeline:
    def analyze(self, image, variant_ids):
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[10:40, 10:40] = True
        localization = Localization(
            status="ok",
            mask=mask,
            bbox=(10, 10, 40, 40),
            timings_ms={"yolo": 1.0, "sam2": 2.0},
        )
        return PipelineOutput(
            localization=localization,
            embeddings={v: np.array([1, 0, 0, 0], dtype=np.float32) for v in variant_ids},
            variant_status={v: "ok" for v in variant_ids},
            timings_ms={
                "yolo": 1.0,
                "sam2": 2.0,
                "preprocess": 3.0,
                "embed": 4.0,
                "total": 10.0,
            },
            embed_ms={v: 4.0 for v in variant_ids},
        )


class FakeEmbeddings:
    def gallery_entries(self, variant_id):
        return [
            GalleryEntry(1, "Milk 1L", 199, np.array([[1, 0, 0, 0]], dtype=np.float32)),
            GalleryEntry(2, "Bread", 250, np.array([[0, 1, 0, 0]], dtype=np.float32)),
        ]


class FakeEmbeddingsNoGallery(FakeEmbeddings):
    def gallery_entries(self, variant_id):
        return []


def jpeg_bytes():
    buffer = BytesIO()
    Image.new("RGB", (64, 64), (100, 100, 100)).save(buffer, format="JPEG")
    return buffer.getvalue()


def make_app(tmp_path, embeddings=None):
    settings = make_settings(tmp_path)
    analysis = AnalysisService(settings, FakePipeline(), embeddings or FakeEmbeddings())
    return create_app(settings, AppState(settings, analysis=analysis))


def test_analyze_returns_top_items_and_annotated_image(tmp_path):
    with TestClient(make_app(tmp_path)) as client:
        response = client.post(
            "/api/analyze",
            files={"image": ("frame.jpg", jpeg_bytes(), "image/jpeg")},
            data={"variants": "v0,v2"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["mask_found"] is True
    assert len(body["variants"]) == 2
    assert body["variants"][0]["top_items"][0]["name"] == "Milk 1L"
    assert body["variants"][0]["top_items"][0]["price_cents"] == 199
    assert body["annotated_jpeg_b64"]


def test_analyze_with_empty_gallery_reports_no_gallery(tmp_path):
    with TestClient(make_app(tmp_path, FakeEmbeddingsNoGallery())) as client:
        body = client.post(
            "/api/analyze",
            files={"image": ("frame.jpg", jpeg_bytes(), "image/jpeg")},
            data={"variants": "v2"},
        ).json()
    assert body["variants"][0]["status"] == "no_gallery"
