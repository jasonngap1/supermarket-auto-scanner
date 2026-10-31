import importlib
import os

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_bootstrap_sets_mps_fallback_before_torch(monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    import app.bootstrap

    importlib.reload(app.bootstrap)
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_health_endpoint(tmp_path):
    settings = Settings(data_dir=tmp_path, warm_models=False)
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["device"]["preference"] == "auto"
    assert body["assets"]["face_detector"] is False
    assert "dinov2" in body["models"]
    assert "sam2" in body["models"]
    assert "sam3" not in body["models"]
