import time
from dataclasses import dataclass, field

import numpy as np

from app.errors import ModelUnavailable
from app.pipeline.types import Localization
from app.pipeline.variants import VARIANTS, preprocess_variant
from app.pipeline.yolo import localize_with_sam, localize_with_yolo_seg

MASK_STATUSES = ("no_detection", "mask_rejected")


@dataclass(frozen=True)
class PipelineOutput:
    localization: Localization
    embeddings: dict[str, np.ndarray]
    variant_status: dict[str, str]
    timings_ms: dict[str, float]
    embed_ms: dict[str, float] = field(default_factory=dict)
    localizations: dict[str, Localization] = field(default_factory=dict)
    variant_timings: dict[str, dict[str, float]] = field(default_factory=dict)


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

        mask_variants = [v for v in variant_ids if VARIANTS[v].crop == "mask"]
        localizations: dict[str, Localization] = {}
        yolo_timings: dict[str, float] = {}

        if mask_variants:
            try:
                yolo_localization = localize_with_yolo_seg(
                    self.settings, self.registry, image_rgb
                )
            except ModelUnavailable:
                yolo_localization = Localization(status="model_error")

            yolo_timings = dict(yolo_localization.timings_ms)
            if "v2" in mask_variants:
                localizations["v2"] = yolo_localization
            if "v3" in mask_variants:
                if yolo_localization.status == "model_error":
                    localizations["v3"] = Localization(status="model_error")
                else:
                    try:
                        localizations["v3"] = localize_with_sam(
                            self.settings,
                            self.registry,
                            image_rgb,
                            yolo_localization.detections,
                            yolo_localization.detection_selection,
                        )
                    except ModelUnavailable:
                        localizations["v3"] = Localization(status="model_error")

        embeddings: dict[str, np.ndarray] = {}
        statuses: dict[str, str] = {}
        embed_ms: dict[str, float] = {}
        variant_timings: dict[str, dict[str, float]] = {}
        preprocess_ms = 0.0

        for variant_id in variant_ids:
            variant = VARIANTS[variant_id]
            localization = localizations.get(variant_id, Localization())
            timings: dict[str, float] = {}
            if variant.crop == "mask":
                timings.update(yolo_timings)
            if variant_id == "v3":
                timings.update(localization.timings_ms)

            if variant.crop == "mask":
                if localization.status == "model_error":
                    statuses[variant_id] = "model_error"
                    variant_timings[variant_id] = timings
                    continue
                if localization.mask is None:
                    statuses[variant_id] = (
                        localization.status
                        if localization.status in MASK_STATUSES
                        else "mask_rejected"
                    )
                    variant_timings[variant_id] = timings
                    continue

            preprocess_start = time.perf_counter()
            crop = preprocess_variant(image_rgb, variant, localization, self.settings)
            preprocess_ms += (time.perf_counter() - preprocess_start) * 1000
            if crop is None:
                statuses[variant_id] = "mask_rejected"
                variant_timings[variant_id] = timings
                continue
            embed_start = time.perf_counter()
            try:
                embeddings[variant_id] = self.registry.embed(crop)
                statuses[variant_id] = "ok"
            except ModelUnavailable:
                statuses[variant_id] = "model_error"
            embed_ms[variant_id] = (time.perf_counter() - embed_start) * 1000
            variant_timings[variant_id] = timings

        primary = Localization()
        if "v2" in variant_ids:
            primary = localizations.get("v2", Localization())
        if primary.mask is None:
            sam_localization = localizations.get("v3")
            if sam_localization is not None and sam_localization.mask is not None:
                primary = sam_localization

        timings = {}
        if mask_variants:
            timings.update(yolo_timings)
            sam2_ms = localizations.get("v3", Localization()).timings_ms.get("sam2")
            if sam2_ms is not None:
                timings["sam2"] = sam2_ms
        timings["preprocess"] = preprocess_ms
        timings["embed"] = sum(embed_ms.values())
        timings["total"] = sum(timings.values())
        return PipelineOutput(
            localization=primary,
            embeddings=embeddings,
            variant_status=statuses,
            timings_ms=timings,
            embed_ms=embed_ms,
            localizations=localizations,
            variant_timings=variant_timings,
        )
