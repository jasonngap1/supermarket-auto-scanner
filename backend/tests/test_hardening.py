from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_scan_rejects_oversized_upload(tmp_path):
    settings = Settings(data_dir=tmp_path, warm_models=False, max_upload_bytes=10)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/scan",
            files={"image": ("frame.jpg", b"x" * 20, "image/jpeg")},
        )
    assert response.status_code == 413


def test_scan_rejects_an_invalid_image(tmp_path):
    settings = Settings(data_dir=tmp_path, warm_models=False)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/scan",
            files={"image": ("frame.jpg", b"not an image", "image/jpeg")},
        )
    assert response.status_code == 422
