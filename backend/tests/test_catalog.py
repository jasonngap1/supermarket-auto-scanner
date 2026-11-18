import json

import numpy as np
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import init_db
from app.main import create_app
from app.services.captures import add_capture
from app.services.catalog import export_catalog, manifest_path
from app.services.items import create_item
from app.services.storage import save_capture_bundle


def make_capture(settings, item):
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    mask = np.ones((32, 32), dtype=bool)
    artifacts = save_capture_bundle(settings, item.id, image, mask)
    return add_capture(
        settings,
        item_id=item.id,
        role="gallery",
        session_id="s1",
        artifacts=artifacts,
        sha256="a" * 64,
        width=32,
        height=32,
        mask_found=True,
        mask_status="ok",
    )


def test_export_then_seed_fresh_database(tmp_path):
    catalog = tmp_path / "catalog"
    source = Settings(data_dir=tmp_path / "source", catalog_dir=catalog)
    init_db(source)
    item = create_item(source, "Milk 1L", 199)
    capture = make_capture(source, item)
    export_catalog(source)

    payload = json.loads(manifest_path(source).read_text())
    assert payload["items"][0]["name"] == "Milk 1L"
    assert payload["captures"][0]["id"] == capture.id
    assert (catalog / "media" / capture.rel_path).is_file()
    assert (catalog / "media" / capture.thumb_rel_path).is_file()

    clone = Settings(
        data_dir=tmp_path / "clone", catalog_dir=catalog, warm_models=False
    )
    with TestClient(create_app(clone)) as client:
        listed = client.get("/api/items").json()
        assert [entry["name"] for entry in listed] == ["Milk 1L"]
        assert listed[0]["gallery_count"] == 1
        assert listed[0]["thumb_rel_path"] == capture.thumb_rel_path

        captures = client.get(
            f"/api/items/{item.id}/captures?role=gallery"
        ).json()
        assert [entry["id"] for entry in captures] == [capture.id]
        assert client.get(f"/media/{capture.thumb_rel_path}").status_code == 200
        assert client.get(f"/media/{capture.rel_path}").status_code == 200


def test_catalog_handles_a_capture_without_a_mask(tmp_path):
    catalog = tmp_path / "catalog"
    source = Settings(data_dir=tmp_path / "source", catalog_dir=catalog)
    init_db(source)
    item = create_item(source, "Milk 1L", 199)
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    artifacts = save_capture_bundle(source, item.id, image)
    capture = add_capture(
        source,
        item_id=item.id,
        role="gallery",
        session_id="s1",
        artifacts=artifacts,
        sha256="b" * 64,
        width=32,
        height=32,
        mask_status="no_detection",
    )
    export_catalog(source)

    payload = json.loads(manifest_path(source).read_text())
    assert payload["captures"][0]["mask_rel_path"] == ""

    clone = Settings(
        data_dir=tmp_path / "clone", catalog_dir=catalog, warm_models=False
    )
    with TestClient(create_app(clone)) as client:
        captures = client.get(
            f"/api/items/{item.id}/captures?role=gallery"
        ).json()
        assert [entry["id"] for entry in captures] == [capture.id]
        assert client.get(f"/media/{capture.thumb_rel_path}").status_code == 200


def test_existing_catalog_refreshes_after_item_changes(tmp_path):
    catalog = tmp_path / "catalog"
    settings = Settings(
        data_dir=tmp_path / "data", catalog_dir=catalog, warm_models=False
    )
    init_db(settings)
    export_catalog(settings)

    with TestClient(create_app(settings)) as client:
        item = client.post(
            "/api/items", json={"name": "Coffee", "price_cents": 450}
        ).json()
        assert json.loads(manifest_path(settings).read_text())["items"][0][
            "name"
        ] == "Coffee"

        assert client.delete(f"/api/items/{item['id']}").status_code == 204
        assert json.loads(manifest_path(settings).read_text())["items"] == []
