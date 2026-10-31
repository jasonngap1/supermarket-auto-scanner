import time

import numpy as np

from app.config import Settings
from app.errors import ModelUnavailable
from app.pipeline.mask_select import select_center_mask
from app.pipeline.preprocess import letterbox
from app.pipeline.prompts import build_center_prompt
from app.pipeline.types import Localization, MaskSelection

EMBED_VARIANT = "v0"
EMBED_MODEL_KEY = "dinov2-small"


class FramePipeline:
    def __init__(self, registry, settings: Settings):
        self.registry = registry
        self.settings = settings

    def localize(self, image_rgb: np.ndarray) -> Localization:
        """Enrollment-only item mask used for the gallery thumbnail."""
        with self.registry.gate():
            return self._localize(image_rgb)

    def _localize(self, image_rgb: np.ndarray) -> Localization:
        timings: dict[str, float] = {}
        start = time.perf_counter()
        try:
            face = self.registry.detect_face(image_rgb)
        except ModelUnavailable:
            # Face rejection is best-effort; the centre point still works.
            face = None
        timings["face"] = (time.perf_counter() - start) * 1000

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
                prompt=prompt,
                selection=selection,
                mask=None,
                status="mask_rejected",
                timings_ms=timings,
                face=face,
            )
        return Localization(
            prompt=prompt,
            selection=selection,
            mask=selection.selected.mask,
            status="ok",
            timings_ms=timings,
            face=face,
        )

    def embed_frame(self, image_rgb: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """Whole-frame DINOv2 embedding; this is the only recognition path."""
        timings: dict[str, float] = {}
        with self.registry.gate():
            start = time.perf_counter()
            crop = letterbox(image_rgb, self.settings.input_size)
            timings["preprocess"] = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            vector = self.registry.embed(crop)
            timings["embed"] = (time.perf_counter() - start) * 1000
        timings["total"] = sum(timings.values())
        return vector, timings
