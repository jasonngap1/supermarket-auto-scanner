from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.dependencies import AppState
from app.main import create_app
from app.pipeline.frame_pipeline import FramePipeline
from app.services.captures import CaptureService
from tests.fakes import FakeRegistry, make_settings


def jpeg_bytes(color=(200, 40, 40)):
    buffer = BytesIO()
    Image.new("RGB", (64, 64), color).save(buffer, format="JPEG")
    return buffer.getvalue()


class FakeEmbeddings:
    def get_capture_vector(self, _capture, _variant):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def make_app(tmp_path, detections=None, embeddings=None):
    settings = make_settings(tmp_path)
    pipeline = FramePipeline(FakeRegistry(detections=detections or []), settings)
    services = AppState(
        settings,
        pipeline=pipeline,
        captures=CaptureService(settings, embeddings),
    )
    return create_app(settings, services)


def test_upload_accepts_any_valid_image_without_localization(tmp_path):
    app = make_app(tmp_path)
    with TestClient(app) as client:
        item = client.post("/api/items", json={"name": "Milk", "price_cents": 199}).json()
        response = client.post(
            f"/api/items/{item['id']}/captures?role=gallery&session_id=G1",
            files={"files": ("frame.jpg", jpeg_bytes(), "image/jpeg")},
        )
        assert response.status_code == 201
        body = response.json()
        assert len(body["accepted"]) == 1
        capture = body["accepted"][0]["capture"]
        assert capture["mask_status"] == "not_localized"
        assert capture["mask_rel_path"] == ""
        assert capture["mask_found"] is False
        assert client.get(f"/media/{capture['thumb_rel_path']}").status_code == 200
        listed = client.get(f"/api/items/{item['id']}/captures?role=gallery").json()
        assert len(listed) == 1


def test_upload_warns_when_test_image_duplicates_gallery(tmp_path):
    app = make_app(tmp_path, embeddings=FakeEmbeddings())
    with TestClient(app) as client:
        item = client.post("/api/items", json={"name": "Milk", "price_cents": 199}).json()
        first = client.post(
            f"/api/items/{item['id']}/captures?role=gallery&session_id=G1",
            files={"files": ("frame.jpg", jpeg_bytes(), "image/jpeg")},
        ).json()
        assert first["accepted"][0]["near_duplicate"] is False
        second = client.post(
            f"/api/items/{item['id']}/captures?role=test&session_id=T1",
            files={"files": ("frame2.jpg", jpeg_bytes((10, 20, 30)), "image/jpeg")},
        ).json()
        assert second["accepted"][0]["near_duplicate"] is True
        assert second["accepted"][0]["near_duplicate_score"] == 1.0
