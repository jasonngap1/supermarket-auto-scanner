from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def make_client(tmp_path):
    return TestClient(create_app(Settings(data_dir=tmp_path, warm_models=False)))


def test_create_list_delete_item_and_persist(tmp_path):
    with make_client(tmp_path) as client:
        created = client.post("/api/items", json={"name": "Milk 1L", "price_cents": 199})
        assert created.status_code == 201
        item = created.json()
        assert item["name"] == "Milk 1L"
        assert item["price_cents"] == 199
        assert item["gallery_count"] == 0

        listed = client.get("/api/items").json()
        assert [i["id"] for i in listed] == [item["id"]]

    with make_client(tmp_path) as client:
        assert client.get("/api/items").json()[0]["name"] == "Milk 1L"

    with make_client(tmp_path) as client:
        deleted = client.delete(f"/api/items/{item['id']}")
        assert deleted.status_code == 204
        assert client.get("/api/items").json() == []


def test_rejects_negative_price(tmp_path):
    with make_client(tmp_path) as client:
        response = client.post("/api/items", json={"name": "Bad", "price_cents": -1})
    assert response.status_code == 422
