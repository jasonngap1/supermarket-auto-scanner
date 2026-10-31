import time
from dataclasses import dataclass, field

import numpy as np

from app.config import Settings
from app.errors import ModelUnavailable
from app.pipeline.mask_select import select_center_mask, select_mask, select_sam3_mask
from app.pipeline.prompts import (
    build_center_prompt,
    build_prompt,
    build_sam3_prompt,
    center_roi_box,
)
from app.pipeline.types import Localization, MaskSelection
from app.pipeline.variants import VARIANTS, preprocess_variant


@dataclass(frozen=True)
class PipelineOutput:
    localization: Localization
    embeddings: dict[str, np.ndarray]
    variant_status: dict[str, str]
    timings_ms: dict[str, float]
    embed_ms: dict[str, float] = field(default_factory=dict)


class FramePipeline:
    def __init__(self, registry, settings: Settings):
        self.registry = registry
        self.settings = settings

    def localize(self, image_rgb: np.ndarray) -> Localization:
        with self.registry.gate():
            return self._localize(image_rgb)

    def _localize(self, image_rgb: np.ndarray) -> Localization:
        timings: dict[str, float] = {}
        start = time.perf_counter()
        hand = None
        try:
            hand = self.registry.detect_hand(image_rgb)
        except ModelUnavailable:
            # Only the hand-prompt SAM2 localizer needs the detector. SAM3 and
            # sam2_center still need a hand for the V1 control, but can run
            # without it.
            if self.settings.localizer == "sam2":
                raise
        timings["hand"] = (time.perf_counter() - start) * 1000

        if self.settings.localizer == "sam3":
            return self._localize_sam3(image_rgb, hand, timings)

        face = None
        if self.settings.localizer == "sam2_center":
            start = time.perf_counter()
            try:
                face = self.registry.detect_face(image_rgb)
            except ModelUnavailable:
                # Face rejection is best-effort; the centre point still works.
                face = None
            timings["face"] = (time.perf_counter() - start) * 1000
            return self._localize_sam2_center(image_rgb, hand, face, timings)
        return self._localize_sam2(image_rgb, hand, timings)

    def _localize_sam2_center(self, image_rgb, hand, face, timings) -> Localization:
        height, width = image_rgb.shape[:2]
        prompt = build_center_prompt(
            width,
            height,
            roi_width=self.settings.center_roi_width,
            roi_height=self.settings.center_roi_height,
            face_bbox=face.bbox if face is not None else None,
        )
        start = time.perf_counter()
        candidates = self.registry.segment(image_rgb, prompt)
        timings["sam2"] = (time.perf_counter() - start) * 1000

        selection = select_center_mask(
            candidates,
            prompt.positive_points[0],
            prompt.box,
            width,
            height,
            face_bbox=face.bbox if face is not None else None,
            min_area_ratio=self.settings.sam2_center_min_area_ratio,
            max_area_ratio=self.settings.sam2_center_max_area_ratio,
            min_pred_iou=self.settings.sam2_center_min_pred_iou,
            max_border_fraction=self.settings.sam2_center_max_border_fraction,
            face_iou_reject=self.settings.sam2_center_face_iou_reject,
            face_coverage_reject=self.settings.sam2_center_face_coverage_reject,
            roi_margin_ratio=self.settings.sam2_center_roi_margin_ratio,
        )
        if selection.selected is None:
            return Localization(
                hand=hand,
                prompt=prompt,
                selection=selection,
                mask=None,
                status="mask_rejected",
                timings_ms=timings,
                face=face,
            )
        return Localization(
            hand=hand,
            prompt=prompt,
            selection=selection,
            mask=selection.selected.mask,
            status="ok",
            timings_ms=timings,
            face=face,
        )

    def _localize_sam2(self, image_rgb, hand, timings) -> Localization:
        if hand is None:
            return Localization(
                hand=None,
                prompt=None,
                selection=MaskSelection(None, [], [], []),
                mask=None,
                status="no_hand",
                timings_ms=timings,
            )

        height, width = image_rgb.shape[:2]
        prompt = build_prompt(
            hand.landmarks_px,
            width,
            height,
            mode=self.settings.prompt_mode,
            pinch_close_ratio=self.settings.pinch_close_ratio,
            box_expand=self.settings.prompt_box_expand,
        )
        start = time.perf_counter()
        candidates = self.registry.segment(image_rgb, prompt)
        timings["sam2"] = (time.perf_counter() - start) * 1000

        selection = select_mask(
            candidates,
            hand.landmarks_px,
            prompt.positive_points[0],
            prompt.negative_points,
            prompt.box,
            width * height,
            min_area_ratio=self.settings.mask_min_area_ratio,
            max_area_ratio=self.settings.mask_max_area_ratio,
            max_landmark_hits=self.settings.mask_max_landmark_hits,
            min_pred_iou=self.settings.mask_min_pred_iou,
        )
        if selection.selected is None:
            return Localization(
                hand=hand,
                prompt=prompt,
                selection=selection,
                mask=None,
                status="mask_rejected",
                timings_ms=timings,
            )
        return Localization(
            hand=hand,
            prompt=prompt,
            selection=selection,
            mask=selection.selected.mask,
            status="ok",
            timings_ms=timings,
        )

    def _localize_sam3(self, image_rgb, hand, timings) -> Localization:
        height, width = image_rgb.shape[:2]
        roi = center_roi_box(
            width,
            height,
            self.settings.center_roi_width,
            self.settings.center_roi_height,
        )
        prompt = build_sam3_prompt(
            width,
            height,
            self.settings.sam3_text_prompt,
            include_box=self.settings.sam3_prompt_box,
            roi_width=self.settings.center_roi_width,
            roi_height=self.settings.center_roi_height,
        )
        start = time.perf_counter()
        candidates = self.registry.segment(image_rgb, prompt)
        timings["sam3"] = (time.perf_counter() - start) * 1000

        selection = select_sam3_mask(
            candidates,
            roi,
            width,
            height,
            min_area_ratio=self.settings.mask_min_area_ratio,
            max_area_ratio=self.settings.mask_max_area_ratio,
            min_score=self.settings.sam3_min_score,
        )
        if selection.selected is None:
            return Localization(
                hand=hand,
                prompt=prompt,
                selection=selection,
                mask=None,
                status="mask_rejected",
                timings_ms=timings,
            )
        return Localization(
            hand=hand,
            prompt=prompt,
            selection=selection,
            mask=selection.selected.mask,
            status="ok",
            timings_ms=timings,
        )

    def analyze(self, image_rgb: np.ndarray, variant_ids: list[str]) -> PipelineOutput:
        with self.registry.gate():
            return self._analyze(image_rgb, variant_ids)

    def _analyze(self, image_rgb: np.ndarray, variant_ids: list[str]) -> PipelineOutput:
        unknown = [v for v in variant_ids if v not in VARIANTS]
        if unknown:
            raise ValueError(f"Unknown variants: {unknown}")
        needs_localization = any(VARIANTS[v].crop in ("hand", "mask") for v in variant_ids)
        localization_error = None
        if needs_localization:
            try:
                localization = self._localize(image_rgb)
            except ModelUnavailable as exc:
                localization_error = str(exc)
                localization = Localization(
                    None, None, MaskSelection(None, [], [], []), None, "model_error"
                )
        else:
            localization = Localization(
                None, None, MaskSelection(None, [], [], []), None, "ok"
            )

        embeddings: dict[str, np.ndarray] = {}
        statuses: dict[str, str] = {}
        embed_ms: dict[str, float] = {}
        preprocess_ms = 0.0

        for variant_id in variant_ids:
            variant = VARIANTS[variant_id]
            if localization_error is not None and variant.crop in ("hand", "mask"):
                statuses[variant_id] = "model_error"
                continue
            if variant.embedder == "clip" and not self.settings.enable_clip:
                statuses[variant_id] = "model_error"
                continue
            if localization.status == "no_hand" and variant.crop in ("hand", "mask"):
                statuses[variant_id] = "no_hand"
                continue
            if variant.crop == "hand" and localization.hand is None:
                statuses[variant_id] = "no_hand"
                continue
            if variant.crop == "mask" and localization.mask is None:
                statuses[variant_id] = "mask_rejected"
                continue
            preprocess_start = time.perf_counter()
            crop = preprocess_variant(image_rgb, variant, localization, self.settings)
            preprocess_ms += (time.perf_counter() - preprocess_start) * 1000
            if crop is None:
                statuses[variant_id] = "no_hand"
                continue
            embed_start = time.perf_counter()
            try:
                embeddings[variant_id] = self.registry.embed(variant.embedder, crop)
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
