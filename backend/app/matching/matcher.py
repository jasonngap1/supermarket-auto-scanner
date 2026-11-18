from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from app.domain import TopItem
from app.matching.vector_index import normalize


@dataclass(frozen=True)
class GalleryEntry:
    item_id: int
    name: str
    price_cents: int
    vectors: Mapping[str, np.ndarray]
    thumb_rel_path: str | None = None


def item_topk(
    queries: Mapping[str, np.ndarray], entries: list[GalleryEntry], k: int = 3
) -> list[TopItem]:
    normalized = {variant: normalize(query) for variant, query in queries.items()}
    scored: list[TopItem] = []
    for entry in entries:
        best: float | None = None
        for variant, query in normalized.items():
            vectors = entry.vectors.get(variant)
            if vectors is None:
                continue
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.size == 0:
                continue
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            scores = (vectors / norms) @ query
            score = float(np.max(scores))
            best = score if best is None else max(best, score)
        if best is None:
            continue
        scored.append(
            TopItem(
                item_id=entry.item_id,
                name=entry.name,
                price_cents=entry.price_cents,
                score=best,
                thumb_rel_path=entry.thumb_rel_path,
            )
        )
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:k]
