from fastapi.testclient import TestClient

from app.dependencies import AppState
from app.main import create_app
from tests.fakes import make_settings


class FakeEvaluationService:
    def start(self, variants, item_ids=None):
        return "run-1"

    def list(self):
        return [{"id": "run-1", "status": "completed"}]

    def get(self, run_id):
        if run_id != "run-1":
            return None
        return {
            "id": "run-1",
            "status": "completed",
            "metrics": {},
            "failures": [{"variant": "v2", "true_item_name": "Milk"}],
        }


def test_start_get_and_failures(tmp_path):
    settings = make_settings(tmp_path)
    services = AppState(settings, evaluation=FakeEvaluationService())
    with TestClient(create_app(settings, services)) as client:
        started = client.post("/api/evaluations", json={"variants": ["v2"]})
        assert started.status_code == 202
        assert started.json()["run_id"] == "run-1"
        assert client.get("/api/evaluations/run-1").status_code == 200
        assert client.get("/api/evaluations/missing").status_code == 404
        failures = client.get("/api/evaluations/run-1/failures?variant=v2").json()
        assert failures[0]["variant"] == "v2"
