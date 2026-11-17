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


def test_updates_name_and_price(tmp_path):
    with make_client(tmp_path) as client:
        item = client.post(
            "/api/items", json={"name": "Milk 1L", "price_cents": 199}
        ).json()

        renamed = client.patch(
            f"/api/items/{item['id']}", json={"name": "Milk 2L"}
        )
        assert renamed.status_code == 200
        assert renamed.json()["name"] == "Milk 2L"
        assert renamed.json()["price_cents"] == 199

        repriced = client.patch(
            f"/api/items/{item['id']}", json={"price_cents": 249}
        )
        assert repriced.status_code == 200
        assert repriced.json()["name"] == "Milk 2L"
        assert repriced.json()["price_cents"] == 249

        assert client.get("/api/items").json()[0]["price_cents"] == 249


def test_update_rejects_invalid_payloads_and_missing_items(tmp_path):
    with make_client(tmp_path) as client:
        item = client.post(
            "/api/items", json={"name": "Milk", "price_cents": 199}
        ).json()
        assert client.patch("/api/items/9999", json={"name": "X"}).status_code == 404
        assert (
            client.patch(f"/api/items/{item['id']}", json={"name": "  "}).status_code
            == 422
        )
        assert (
            client.patch(
                f"/api/items/{item['id']}", json={"price_cents": -1}
            ).status_code
            == 422
        )
        assert client.patch(f"/api/items/{item['id']}", json={}).status_code == 422
