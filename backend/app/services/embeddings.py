import numpy as np

from app.matching.matcher import GalleryEntry
from app.pipeline.frame_pipeline import EMBED_MODEL_KEY
from app.pipeline.preprocess import (
    EMBED_VARIANTS,
    EMBED_VARIANT,
    preprocess_for_variant,
    preprocess_version_for_variant,
)
from app.services.captures import get_embedding, list_gallery_captures, upsert_embedding
from app.services.storage import load_image_rgb


class EmbeddingService:
    def __init__(self, settings, registry):
        self.settings = settings
        self.registry = registry

    def get_capture_vector(
        self, capture, variant: str = EMBED_VARIANT
    ) -> np.ndarray | None:
        version = preprocess_version_for_variant(variant)
        cached = get_embedding(
            self.settings,
            capture.id,
            variant,
            EMBED_MODEL_KEY,
            version,
        )
        if cached is not None:
            return cached

        image = load_image_rgb(self.settings, capture.rel_path)
        crop = preprocess_for_variant(image, variant, self.settings.input_size)
        vector = self.registry.embed(crop)
        upsert_embedding(
            self.settings,
            capture.id,
            variant,
            EMBED_MODEL_KEY,
            version,
            vector,
        )
        return vector

    def gallery_entries(self) -> list[GalleryEntry]:
        grouped: dict[int, dict] = {}
        for capture, name, price_cents in list_gallery_captures(self.settings):
            bucket = grouped.setdefault(
                capture.item_id,
                {
                    "name": name,
                    "price": price_cents,
                    "vectors": {variant: [] for variant in EMBED_VARIANTS},
                    "thumb": capture.thumb_rel_path,
                },
            )
            for variant in EMBED_VARIANTS:
                vector = self.get_capture_vector(capture, variant)
                if vector is not None:
                    bucket["vectors"][variant].append(vector)
        return [
            GalleryEntry(
                item_id,
                data["name"],
                data["price"],
                {
                    variant: np.vstack(vectors)
                    for variant, vectors in data["vectors"].items()
                    if vectors
                },
                data["thumb"],
            )
            for item_id, data in grouped.items()
        ]
