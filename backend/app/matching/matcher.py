from dataclasses import dataclass

import numpy as np

from app.domain import TopItem
from app.matching.vector_index import normalize


@dataclass(frozen=True)
class GalleryEntry:
    item_id: int
    name: str
    price_cents: int
    vectors: np.ndarray
    thumb_rel_path: str | None = None


def item_topk(query: np.ndarray, entries: list[GalleryEntry], k: int = 3) -> list[TopItem]:
    query = normalize(query)
    scored: list[TopItem] = []
    for entry in entries:
        vectors = np.asarray(entry.vectors, dtype=np.float32)
        if vectors.size == 0:
            continue
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        scores = (vectors / norms) @ query
        scored.append(
            TopItem(
                item_id=entry.item_id,
                name=entry.name,
                price_cents=entry.price_cents,
                score=float(np.max(scores)),
                thumb_rel_path=entry.thumb_rel_path,
            )
        )
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:k]
