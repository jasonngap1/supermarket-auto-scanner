import numpy as np
import pytest

from app.matching.matcher import GalleryEntry, item_topk
from app.matching.vector_index import ExactCosineIndex, normalize


def test_exact_index_ranks_by_cosine():
    index = ExactCosineIndex()
    index.add([10, 20], np.array([[1, 0], [0, 1]], dtype=np.float32))
    assert index.search(np.array([0.9, 0.1]), k=2) == [
        (10, pytest.approx(0.9938837)),
        (20, pytest.approx(0.11043153)),
    ]


def test_normalize_is_safe_for_zero_vector():
    assert np.allclose(normalize(np.zeros(3, dtype=np.float32)), 0)


def test_item_topk_takes_max_over_gallery_angles_and_variants():
    entries = [
        GalleryEntry(
            1,
            "Milk",
            199,
            {
                "v0": np.array([[1, 0], [0.8, 0.6]], dtype=np.float32),
                "center": np.array([[0.6, 0.8]], dtype=np.float32),
            },
        ),
        GalleryEntry(
            2,
            "Bread",
            250,
            {"v0": np.array([[0, 1]], dtype=np.float32)},
        ),
    ]
    queries = {
        "v0": np.array([1, 0], dtype=np.float32),
        "center": np.array([0, 1], dtype=np.float32),
    }
    top = item_topk(queries, entries, k=3)
    assert [t.item_id for t in top] == [1, 2]
    assert top[0].score == pytest.approx(1.0)
    assert len(item_topk(queries, entries, k=5)) == 2


def test_empty_gallery_returns_empty():
    assert item_topk({"v0": np.array([1, 0], dtype=np.float32)}, [], k=3) == []
