from typing import Protocol

import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


class VectorIndex(Protocol):
    def add(self, ids: list[int], vectors: np.ndarray) -> None: ...
    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]: ...


class ExactCosineIndex:
    """Exact cosine search; replace with an HNSW implementation when N grows."""

    def __init__(self) -> None:
        self._ids: list[int] = []
        self._matrix = np.zeros((0, 0), dtype=np.float32)

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        self._ids.extend(ids)
        self._matrix = matrix if self._matrix.size == 0 else np.vstack([self._matrix, matrix])

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        if not self._ids:
            return []
        scores = self._matrix @ normalize(query)
        order = np.argsort(-scores)[:k]
        return [(self._ids[int(i)], float(scores[int(i)])) for i in order]
