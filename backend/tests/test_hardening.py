from fastapi.testclient import TestClient

from app.config import Settings
from app.db import db, init_db
from app.main import create_app
from app.services.evaluation import recover_stale_runs


def test_analyze_rejects_oversized_upload(tmp_path):
    settings = Settings(data_dir=tmp_path, warm_models=False, max_upload_bytes=10)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/analyze",
            files={"image": ("frame.jpg", b"x" * 20, "image/jpeg")},
            data={"variants": "v0"},
        )
    assert response.status_code == 413


def test_stale_runs_are_failed_on_restart(tmp_path):
    settings = Settings(data_dir=tmp_path, warm_models=False)
    init_db(settings)
    with db(settings) as connection:
        connection.execute(
            "INSERT INTO eval_runs (id, created_at, status, request_json) "
            "VALUES ('stale', 'now', 'running', '{}')"
        )
    recover_stale_runs(settings)
    with db(settings) as connection:
        row = connection.execute(
            "SELECT status, error FROM eval_runs WHERE id='stale'"
        ).fetchone()
    assert row["status"] == "failed"
    assert "interrupted" in row["error"]
