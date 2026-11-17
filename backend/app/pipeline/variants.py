from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.config import Settings
from app.pipeline.preprocess import (
    PREPROCESS_VERSION,
    letterbox,
    normalize_mask_crop,
)
from app.pipeline.types import Localization

MODEL_KEY = "dinov2-small"


@dataclass(frozen=True)
class Variant:
    id: str
    label: str
    crop: Literal["whole", "mask"]


VARIANTS: dict[str, Variant] = {
    "v0": Variant("v0", "Whole frame + DINOv2", "whole"),
    "v2": Variant("v2", "YOLO seg mask + DINOv2", "mask"),
    "v3": Variant("v3", "YOLO box + SAM2 mask + DINOv2", "mask"),
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
        crop = normalize_mask_crop(image_rgb, localization.mask, margin_ratio=0.10)
    return letterbox(crop, settings.input_size)


def embedding_preprocess_version(variant: Variant, settings: Settings) -> str:
    if variant.crop == "whole":
        return PREPROCESS_VERSION
    version = (
        f"{PREPROCESS_VERSION}+mask-norm-v1+yolo-seg-v1"
        f":{settings.yolo_seg_model_id}"
        f":conf{settings.yolo_conf}:iou{settings.yolo_iou}:sz{settings.yolo_imgsz}"
        f":allow{settings.yolo_allowed_classes}:reject{settings.yolo_reject_classes}"
        f":area{settings.yolo_min_area_ratio}-{settings.yolo_max_area_ratio}"
        f":center{settings.yolo_center_weight}"
        f":box{settings.mask_min_box_iou}"
        f":cover{settings.mask_min_box_coverage}"
        f":border{settings.mask_max_border_fraction}"
    )
    if variant.id == "v3":
        version += (
            f":sam2{settings.sam2_model_id}"
            f":pred{settings.sam_min_pred_iou}"
        )
    return version
