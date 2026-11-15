from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.config import Settings
from app.pipeline.preprocess import PREPROCESS_VERSION, crop_mask, letterbox
from app.pipeline.types import Localization

MODEL_KEY = "dinov2-small"


@dataclass(frozen=True)
class Variant:
    id: str
    label: str
    crop: Literal["whole", "mask"]


VARIANTS: dict[str, Variant] = {
    "v0": Variant("v0", "Whole frame + DINOv2", "whole"),
    "v2": Variant("v2", "YOLO + SAM mask + DINOv2", "mask"),
}


def preprocess_variant(
    image_rgb: np.ndarray,
    variant: Variant,
    localization: Localization,
    settings: Settings,
) -> np.ndarray | None:
    if variant.crop == "whole":
        crop = image_rgb
    else:
        if localization.mask is None:
            return None
        crop = crop_mask(image_rgb, localization.mask, margin_ratio=0.10)
    return letterbox(crop, settings.input_size)


def embedding_preprocess_version(variant: Variant, settings: Settings) -> str:
    if variant.crop == "whole":
        return PREPROCESS_VERSION
    return (
        f"{PREPROCESS_VERSION}+yolo-sam-v1:{settings.yolo_det_model_id}"
        f":conf{settings.yolo_conf}:iou{settings.yolo_iou}:sz{settings.yolo_imgsz}"
        f":allow{settings.yolo_allowed_classes}:reject{settings.yolo_reject_classes}"
        f":area{settings.yolo_min_area_ratio}-{settings.yolo_max_area_ratio}"
        f":center{settings.yolo_center_weight}"
        f":sam2{settings.sam2_model_id}"
        f":pred{settings.yolo_sam_min_pred_iou}"
        f":box{settings.yolo_sam_min_box_iou}"
        f":cover{settings.yolo_sam_min_box_coverage}"
        f":border{settings.yolo_sam_max_border_fraction}"
    )
