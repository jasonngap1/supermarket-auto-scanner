from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.dependencies import AppState
from app.main import create_app
from app.pipeline.frame_pipeline import FramePipeline
from app.services.captures import CaptureService
from tests.fakes import FakeRegistry, FakeSam2, make_settings


def jpeg_bytes(color=(200, 40, 40)):
    buffer = BytesIO()
    Image.new("RGB", (64, 64), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def make_app(tmp_path, segmenter=None):
    settings = make_settings(tmp_path)
    pipeline = FramePipeline(FakeRegistry(segmenter=segmenter), settings)
    services = AppState(
        settings,
        pipeline=pipeline,
        captures=CaptureService(settings, pipeline),
    )
    return create_app(settings, services)


def test_upload_accepts_gallery_and_sets_item_thumbnail(tmp_path):
    app = make_app(tmp_path)
    with TestClient(app) as client:
        item = client.post("/api/items", json={"name": "Milk", "price_cents": 199}).json()
        assert item["thumb_rel_path"] is None
        response = client.post(
            f"/api/items/{item['id']}/captures?role=gallery&session_id=G1",
            files={"files": ("frame.jpg", jpeg_bytes(), "image/jpeg")},
        )
        assert response.status_code == 201
        body = response.json()
        assert len(body["accepted"]) == 1
        capture = body["accepted"][0]
        assert capture["mask_status"] == "ok"
        assert client.get(f"/media/{capture['thumb_rel_path']}").status_code == 200

        listed = client.get(f"/api/items/{item['id']}/captures?role=gallery").json()
        assert len(listed) == 1
        updated = client.get("/api/items").json()[0]
        assert updated["gallery_count"] == 1
        assert updated["thumb_rel_path"] == capture["thumb_rel_path"]


def test_only_gallery_role_is_accepted(tmp_path):
    app = make_app(tmp_path)
    with TestClient(app) as client:
        item = client.post("/api/items", json={"name": "Milk", "price_cents": 199}).json()
        response = client.post(
            f"/api/items/{item['id']}/captures?role=test",
            files={"files": ("frame.jpg", jpeg_bytes(), "image/jpeg")},
        )
    assert response.status_code == 422


def test_upload_rejects_a_missing_centre_mask(tmp_path):
    app = make_app(tmp_path, segmenter=FakeSam2(empty=True))
    with TestClient(app) as client:
        item = client.post("/api/items", json={"name": "Milk", "price_cents": 199}).json()
        response = client.post(
            f"/api/items/{item['id']}/captures?role=gallery&session_id=G1",
            files={"files": ("frame.jpg", jpeg_bytes(), "image/jpeg")},
        )
    assert response.status_code == 201
    assert response.json()["accepted"] == []
    assert response.json()["rejected"][0]["reason"] == "mask_rejected"
