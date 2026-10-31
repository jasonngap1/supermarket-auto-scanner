from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.config import Settings
from app.pipeline.preprocess import crop_box, crop_mask, letterbox
from app.pipeline.prompts import hand_crop_box
from app.pipeline.types import Localization

MODEL_KEYS = {"dinov2": "dinov2-small", "clip": "clip-vit-b32"}


@dataclass(frozen=True)
class Variant:
    id: str
    label: str
    crop: Literal["whole", "hand", "mask"]
    embedder: Literal["dinov2", "clip"]


VARIANTS: dict[str, Variant] = {
    "v0": Variant("v0", "Whole frame + DINOv2", "whole", "dinov2"),
    "v1": Variant("v1", "Hand crop + DINOv2", "hand", "dinov2"),
    "v2": Variant("v2", "Mask crop + DINOv2", "mask", "dinov2"),
    "v3": Variant("v3", "Mask crop + CLIP", "mask", "clip"),
}


def preprocess_variant(
    image_rgb: np.ndarray,
    variant: Variant,
    localization: Localization,
    settings: Settings,
) -> np.ndarray | None:
    height, width = image_rgb.shape[:2]
    if variant.crop == "whole":
        crop = image_rgb
    elif variant.crop == "hand":
        if localization.hand is None:
            return None
        box = hand_crop_box(
            localization.hand.landmarks_px, width, height, settings.prompt_box_expand
        )
        crop = crop_box(image_rgb, box, margin_ratio=0.0)
    else:
        if localization.mask is None:
            return None
        crop = crop_mask(image_rgb, localization.mask, margin_ratio=0.10)
    return letterbox(crop, settings.input_size)
