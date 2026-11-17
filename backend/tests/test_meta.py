from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_meta_lists_variants_and_currency(tmp_path):
    settings = Settings(data_dir=tmp_path, warm_models=False, currency_symbol="$")
    with TestClient(create_app(settings)) as client:
        body = client.get("/api/meta").json()
    assert body["currency_symbol"] == "$"
    assert [v["id"] for v in body["variants"]] == ["v0", "v2", "v3"]
    assert body["variants"][1]["crop"] == "mask"
    assert body["variants"][2]["crop"] == "mask"
