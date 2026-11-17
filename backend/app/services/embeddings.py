import time
from dataclasses import dataclass

import numpy as np

from app.errors import ModelUnavailable
from app.matching.matcher import GalleryEntry
from app.pipeline.types import Localization
from app.pipeline.variants import (
    MODEL_KEY,
    VARIANTS,
    embedding_preprocess_version,
    preprocess_variant,
)
from app.pipeline.yolo import localize_with_yolo_seg
from app.services.captures import (
    get_embedding,
    list_gallery_captures,
    upsert_embedding,
)
from app.services.storage import load_image_rgb


@dataclass(frozen=True)
class VectorResult:
    vector: np.ndarray | None
    status: str
    timings_ms: dict[str, float]


class EmbeddingService:
    def __init__(self, settings, registry):
        self.settings = settings
        self.registry = registry

    def get_capture_vector_details(self, capture, variant_id: str) -> VectorResult:
        variant = VARIANTS[variant_id]
        version = embedding_preprocess_version(variant, self.settings)
        lookup_start = time.perf_counter()
        cached = get_embedding(
            self.settings, capture.id, variant_id, MODEL_KEY, version
        )
        lookup_ms = (time.perf_counter() - lookup_start) * 1000
        if cached is not None:
            return VectorResult(cached, "ok", {"lookup": lookup_ms})

        image = load_image_rgb(self.settings, capture.rel_path)
        localization = Localization()
        timings: dict[str, float] = {}
        if variant.crop == "mask":
            try:
                localization = localize_with_yolo_seg(
                    self.settings, self.registry, image
                )
            except ModelUnavailable:
                return VectorResult(None, "model_error", timings)
            timings.update(localization.timings_ms)
            if localization.status != "ok" or localization.mask is None:
                status = (
                    localization.status
                    if localization.status in ("no_detection", "mask_rejected")
                    else "mask_rejected"
                )
                return VectorResult(None, status, timings)

        preprocess_start = time.perf_counter()
        crop = preprocess_variant(image, variant, localization, self.settings)
        timings["preprocess"] = (time.perf_counter() - preprocess_start) * 1000
        if crop is None:
            return VectorResult(None, "mask_rejected", timings)

        embed_start = time.perf_counter()
        try:
            vector = self.registry.embed(crop)
        except ModelUnavailable:
            return VectorResult(None, "model_error", timings)
        timings["embed"] = (time.perf_counter() - embed_start) * 1000
        upsert_embedding(
            self.settings, capture.id, variant_id, MODEL_KEY, version, vector
        )
        return VectorResult(vector, "ok", timings)

    def get_capture_vector(self, capture, variant_id: str) -> np.ndarray | None:
        return self.get_capture_vector_details(capture, variant_id).vector

    def gallery_entries(self, variant_id: str = "v2") -> list[GalleryEntry]:
        grouped: dict[int, dict] = {}
        for capture, name, price_cents in list_gallery_captures(self.settings):
            vector = self.get_capture_vector(capture, variant_id)
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
                {variant_id: np.vstack(data["vectors"])},
                data["thumb"],
            )
            for item_id, data in grouped.items()
        ]
