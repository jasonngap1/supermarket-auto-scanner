import numpy as np

from app.matching.matcher import GalleryEntry
from app.pipeline.preprocess import PREPROCESS_VERSION
from app.pipeline.variants import MODEL_KEYS, VARIANTS, preprocess_variant
from app.services.captures import (
    get_embedding,
    list_gallery_captures,
    load_capture_localization,
    upsert_embedding,
)
from app.services.storage import load_image_rgb


class EmbeddingService:
    def __init__(self, settings, registry):
        self.settings = settings
        self.registry = registry

    def get_capture_vector(self, capture, variant_id: str) -> np.ndarray | None:
        variant = VARIANTS[variant_id]
        model_key = MODEL_KEYS[variant.embedder]
        cached = get_embedding(
            self.settings, capture.id, variant_id, model_key, PREPROCESS_VERSION
        )
        if cached is not None:
            return cached

        image = load_image_rgb(self.settings, capture.rel_path)
        localization = load_capture_localization(self.settings, capture)
        crop = preprocess_variant(image, variant, localization, self.settings)
        if crop is None:
            return None
        vector = self.registry.embed(variant.embedder, crop)
        upsert_embedding(
            self.settings, capture.id, variant_id, model_key, PREPROCESS_VERSION, vector
        )
        return vector

    def gallery_entries(self, variant_id: str) -> list[GalleryEntry]:
        grouped: dict[int, dict] = {}
        for capture, name, price_cents in list_gallery_captures(self.settings):
            vector = self.get_capture_vector(capture, variant_id)
            if vector is None:
                continue
            bucket = grouped.setdefault(
                capture.item_id, {"name": name, "price": price_cents, "vectors": []}
            )
            bucket["vectors"].append(vector)
        return [
            GalleryEntry(item_id, data["name"], data["price"], np.vstack(data["vectors"]))
            for item_id, data in grouped.items()
        ]
