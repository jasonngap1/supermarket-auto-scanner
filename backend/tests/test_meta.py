from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_meta_only_exposes_currency(tmp_path):
    settings = Settings(data_dir=tmp_path, warm_models=False, currency_symbol="$")
    with TestClient(create_app(settings)) as client:
        body = client.get("/api/meta").json()
    assert body == {"currency_symbol": "$"}
