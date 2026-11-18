import time
from dataclasses import dataclass, field

import numpy as np

from app.errors import ModelUnavailable
from app.pipeline.types import Localization
from app.pipeline.variants import VARIANTS, preprocess_variant
from app.pipeline.yolo import localize_with_yolo_seg


@dataclass(frozen=True)
class PipelineOutput:
    localization: Localization
    embeddings: dict[str, np.ndarray]
    variant_status: dict[str, str]
    timings_ms: dict[str, float]
    embed_ms: dict[str, float] = field(default_factory=dict)


class FramePipeline:
    def __init__(self, registry, settings):
        self.registry = registry
        self.settings = settings

    def analyze(self, image_rgb: np.ndarray, variant_ids: list[str]) -> PipelineOutput:
        with self.registry.gate():
            return self._analyze(image_rgb, variant_ids)

    def _analyze(self, image_rgb: np.ndarray, variant_ids: list[str]) -> PipelineOutput:
        unknown = [v for v in variant_ids if v not in VARIANTS]
        if unknown:
            raise ValueError(f"Unknown variants: {unknown}")

        needs_mask = any(VARIANTS[v].crop == "mask" for v in variant_ids)
        localization = Localization()
        if needs_mask:
            try:
                localization = localize_with_yolo_seg(
                    self.settings, self.registry, image_rgb
                )
            except ModelUnavailable:
                localization = Localization(status="model_error")

        embeddings: dict[str, np.ndarray] = {}
        statuses: dict[str, str] = {}
        embed_ms: dict[str, float] = {}
        preprocess_ms = 0.0

        for variant_id in variant_ids:
            variant = VARIANTS[variant_id]
            if variant.crop == "mask":
                if localization.status == "model_error":
                    statuses[variant_id] = "model_error"
                    continue
                if localization.mask is None:
                    statuses[variant_id] = (
                        localization.status
                        if localization.status in ("no_detection", "mask_rejected")
                        else "mask_rejected"
                    )
                    continue
            preprocess_start = time.perf_counter()
            crop = preprocess_variant(image_rgb, variant, localization, self.settings)
            preprocess_ms += (time.perf_counter() - preprocess_start) * 1000
            if crop is None:
                statuses[variant_id] = "mask_rejected"
                continue
            embed_start = time.perf_counter()
            try:
                embeddings[variant_id] = self.registry.embed(crop)
                statuses[variant_id] = "ok"
            except ModelUnavailable:
                statuses[variant_id] = "model_error"
            embed_ms[variant_id] = (time.perf_counter() - embed_start) * 1000

        timings = dict(localization.timings_ms)
        timings["preprocess"] = preprocess_ms
        timings["embed"] = sum(embed_ms.values())
        timings["total"] = sum(timings.values())
        return PipelineOutput(
            localization=localization,
            embeddings=embeddings,
            variant_status=statuses,
            timings_ms=timings,
            embed_ms=embed_ms,
        )
