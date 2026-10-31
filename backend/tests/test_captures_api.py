from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.dependencies import AppState
from app.main import create_app
from app.pipeline.frame_pipeline import FramePipeline
from app.services.captures import CaptureService
from tests.fakes import FakeRegistry, FakeSam2, NoHand, make_settings


def jpeg_bytes(color=(200, 40, 40)):
    buffer = BytesIO()
    Image.new("RGB", (64, 64), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def make_app(tmp_path, hand=None, localizer="sam3", segmenter=None):
    settings = make_settings(tmp_path, localizer=localizer)
    pipeline = FramePipeline(FakeRegistry(hand, segmenter), settings)
    services = AppState(
        settings,
        pipeline=pipeline,
        captures=CaptureService(settings, pipeline),
    )
    return create_app(settings, services)


def test_upload_accepts_valid_capture_and_serves_media(tmp_path):
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
        capture = body["accepted"][0]
        assert capture["mask_status"] == "ok"
        assert client.get(f"/media/{capture['thumb_rel_path']}").status_code == 200
        listed = client.get(f"/api/items/{item['id']}/captures?role=gallery").json()
        assert len(listed) == 1


def test_upload_accepts_centred_item_without_hand(tmp_path):
    app = make_app(tmp_path, hand=NoHand())
    with TestClient(app) as client:
        item = client.post("/api/items", json={"name": "Milk", "price_cents": 199}).json()
        response = client.post(
            f"/api/items/{item['id']}/captures?role=gallery&session_id=G1",
            files={"files": ("frame.jpg", jpeg_bytes(), "image/jpeg")},
        )
        assert response.status_code == 201
        body = response.json()
        assert len(body["accepted"]) == 1
        assert body["accepted"][0]["capture"]["hand_found"] is False
        assert body["accepted"][0]["mask_status"] == "ok"


def test_upload_accepts_centre_prompt_without_hand(tmp_path):
    app = make_app(
        tmp_path, hand=NoHand(), localizer="sam2_center", segmenter=FakeSam2()
    )
    with TestClient(app) as client:
        item = client.post("/api/items", json={"name": "Milk", "price_cents": 199}).json()
        response = client.post(
            f"/api/items/{item['id']}/captures?role=gallery&session_id=G1",
            files={"files": ("frame.jpg", jpeg_bytes(), "image/jpeg")},
        )
        assert response.status_code == 201
        body = response.json()
        assert len(body["accepted"]) == 1
        assert body["accepted"][0]["capture"]["hand_found"] is False
        assert body["accepted"][0]["mask_status"] == "ok"


def test_upload_rejects_no_hand_in_sam2_mode(tmp_path):
    app = make_app(tmp_path, hand=NoHand(), localizer="sam2", segmenter=FakeSam2())
    with TestClient(app) as client:
        item = client.post("/api/items", json={"name": "Milk", "price_cents": 199}).json()
        response = client.post(
            f"/api/items/{item['id']}/captures?role=gallery&session_id=G1",
            files={"files": ("frame.jpg", jpeg_bytes(), "image/jpeg")},
        )
        assert response.status_code == 201
        assert response.json()["accepted"] == []
        assert response.json()["rejected"][0]["reason"] == "no_hand"
        assert client.get(f"/api/items/{item['id']}/captures").json() == []
