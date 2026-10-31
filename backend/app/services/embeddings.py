import numpy as np

from app.matching.matcher import GalleryEntry
from app.pipeline.frame_pipeline import EMBED_MODEL_KEY, EMBED_VARIANT
from app.pipeline.preprocess import PREPROCESS_VERSION, letterbox
from app.services.captures import get_embedding, list_gallery_captures, upsert_embedding
from app.services.storage import load_image_rgb


class EmbeddingService:
    def __init__(self, settings, registry):
        self.settings = settings
        self.registry = registry

    def get_capture_vector(self, capture) -> np.ndarray | None:
        cached = get_embedding(
            self.settings,
            capture.id,
            EMBED_VARIANT,
            EMBED_MODEL_KEY,
            PREPROCESS_VERSION,
        )
        if cached is not None:
            return cached

        image = load_image_rgb(self.settings, capture.rel_path)
        crop = letterbox(image, self.settings.input_size)
        vector = self.registry.embed(crop)
        upsert_embedding(
            self.settings,
            capture.id,
            EMBED_VARIANT,
            EMBED_MODEL_KEY,
            PREPROCESS_VERSION,
            vector,
        )
        return vector

    def gallery_entries(self) -> list[GalleryEntry]:
        grouped: dict[int, dict] = {}
        for capture, name, price_cents in list_gallery_captures(self.settings):
            vector = self.get_capture_vector(capture)
            if vector is None:
                continue
            bucket = grouped.setdefault(
                capture.item_id,
                {
                    "name": name,
                    "price": price_cents,
                    "vectors": [],
                    "thumb": capture.thumb_rel_path,
                },
            )
            bucket["vectors"].append(vector)
        return [
            GalleryEntry(
                item_id,
                data["name"],
                data["price"],
                np.vstack(data["vectors"]),
                data["thumb"],
            )
            for item_id, data in grouped.items()
        ]
