from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.dependencies import AppState
from app.main import create_app
from app.pipeline.frame_pipeline import FramePipeline
from app.services.captures import CaptureService
from tests.fakes import FakeRegistry, detection, make_settings


def jpeg_bytes(color=(200, 40, 40)):
    buffer = BytesIO()
    Image.new("RGB", (100, 100), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def make_app(tmp_path, detections=None):
    settings = make_settings(tmp_path)
    registry = FakeRegistry(detections=detections or [])
    pipeline = FramePipeline(registry, settings)
    services = AppState(
        settings,
        pipeline=pipeline,
        captures=CaptureService(settings, registry),
    )
    return create_app(settings, services)


def media_files(tmp_path):
    media = tmp_path / "media"
    if not media.exists():
        return []
    return [path for path in media.rglob("*") if path.is_file()]


def test_upload_stores_the_yolo_mask_and_thumbnail(tmp_path):
    app = make_app(tmp_path, detections=[detection()])
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
        assert capture["mask_rel_path"]
        assert capture["capture"]["mask_found"] is True
        assert capture["capture"]["mask_status"] == "ok"
        assert client.get(f"/media/{capture['thumb_rel_path']}").status_code == 200
        assert client.get(f"/media/{capture['mask_rel_path']}").status_code == 200

        listed = client.get(f"/api/items/{item['id']}/captures?role=gallery").json()
        assert len(listed) == 1
        updated = client.get("/api/items").json()[0]
        assert updated["gallery_count"] == 1
        assert updated["thumb_rel_path"] == capture["thumb_rel_path"]


def test_upload_rejects_a_frame_without_a_detection(tmp_path):
    app = make_app(tmp_path)
    with TestClient(app) as client:
        item = client.post("/api/items", json={"name": "Milk", "price_cents": 199}).json()
        body = client.post(
            f"/api/items/{item['id']}/captures?role=gallery&session_id=G1",
            files={"files": ("frame.jpg", jpeg_bytes(), "image/jpeg")},
        ).json()
        listed = client.get(f"/api/items/{item['id']}/captures?role=gallery").json()
    assert body["accepted"] == []
    assert body["rejected"][0]["reason"] == "no_detection"
    assert listed == []
    assert media_files(tmp_path) == []


def test_upload_rejects_a_detection_without_a_mask(tmp_path):
    app = make_app(tmp_path, detections=[detection(with_mask=False)])
    with TestClient(app) as client:
        item = client.post("/api/items", json={"name": "Milk", "price_cents": 199}).json()
        body = client.post(
            f"/api/items/{item['id']}/captures?role=gallery&session_id=G1",
            files={"files": ("frame.jpg", jpeg_bytes(), "image/jpeg")},
        ).json()
        listed = client.get(f"/api/items/{item['id']}/captures?role=gallery").json()
    assert body["accepted"] == []
    assert body["rejected"][0]["reason"] == "mask_rejected"
    assert listed == []
    assert media_files(tmp_path) == []


def test_only_gallery_role_is_accepted(tmp_path):
    app = make_app(tmp_path)
    with TestClient(app) as client:
        item = client.post("/api/items", json={"name": "Milk", "price_cents": 199}).json()
        response = client.post(
            f"/api/items/{item['id']}/captures?role=test",
            files={"files": ("frame.jpg", jpeg_bytes(), "image/jpeg")},
        )
    assert response.status_code == 422
