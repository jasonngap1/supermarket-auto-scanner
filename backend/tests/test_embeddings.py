from types import SimpleNamespace

import numpy as np

from app.config import Settings
from app.services import embeddings as embeddings_module
from app.services.embeddings import EmbeddingService
from tests.fakes import FakeRegistry, detection


def service_and_cache(tmp_path, monkeypatch, registry):
    settings = Settings(data_dir=tmp_path, warm_models=False)
    cache: dict[tuple, np.ndarray] = {}
    versions: list[str] = []

    def fake_get(_settings, capture_id, variant, model_key, version):
        versions.append(version)
        return cache.get((capture_id, variant, model_key, version))

    def fake_upsert(_settings, capture_id, variant, model_key, version, vector):
        cache[(capture_id, variant, model_key, version)] = vector

    monkeypatch.setattr(embeddings_module, "get_embedding", fake_get)
    monkeypatch.setattr(embeddings_module, "upsert_embedding", fake_upsert)
    monkeypatch.setattr(
        embeddings_module,
        "load_image_rgb",
        lambda _settings, _path: np.full((100, 100, 3), 128, dtype=np.uint8),
    )
    return EmbeddingService(settings, registry), versions


def test_v2_detects_once_and_reuses_cache(tmp_path, monkeypatch):
    registry = FakeRegistry(detections=[detection()])
    service, versions = service_and_cache(tmp_path, monkeypatch, registry)
    capture = SimpleNamespace(id=1, rel_path="image.jpg")

    first = service.get_capture_vector_details(capture, "v2")
    second = service.get_capture_vector_details(capture, "v2")

    assert first.vector is not None
    assert second.vector is not None
    assert np.array_equal(first.vector, second.vector)
    assert registry.detect_calls == 1
    assert versions[0] == versions[1]
    assert "yolo-seg-v1" in versions[0]
    assert "mask-norm-v1" in versions[0]


def test_v0_uses_a_different_cache_version_and_no_detection(tmp_path, monkeypatch):
    registry = FakeRegistry(detections=[detection()])
    service, versions = service_and_cache(tmp_path, monkeypatch, registry)
    capture = SimpleNamespace(id=1, rel_path="image.jpg")

    service.get_capture_vector_details(capture, "v2")
    service.get_capture_vector_details(capture, "v0")

    assert versions[-1] != versions[0]
    assert registry.detect_calls == 1


def test_v2_reports_no_detection_without_caching(tmp_path, monkeypatch):
    registry = FakeRegistry(detections=[])
    service, _versions = service_and_cache(tmp_path, monkeypatch, registry)
    capture = SimpleNamespace(id=1, rel_path="image.jpg")

    result = service.get_capture_vector_details(capture, "v2")
    assert result.vector is None
    assert result.status == "no_detection"
    assert service.get_capture_vector(capture, "v2") is None
    assert registry.detect_calls == 2
