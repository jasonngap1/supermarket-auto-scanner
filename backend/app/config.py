from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AUTO_SCANNER_", env_file=".env", extra="ignore"
    )

    data_dir: Path = PROJECT_ROOT / "data"
    device: str = "auto"  # auto | mps | cpu
    currency_symbol: str = "$"
    warm_models: bool = True
    enable_clip: bool = False

    localizer: Literal["sam3", "sam2", "sam2_center"] = "sam2_center"
    sam2_model_id: str = "facebook/sam2.1-hiera-tiny"
    sam3_model_id: str = "vil-uob/sam3-litetext-s0"
    dinov2_model_id: str = "facebook/dinov2-small"
    clip_model_id: str = "openai/clip-vit-base-patch32"
    hand_asset_path: Path | None = None
    face_asset_path: Path | None = None

    input_size: int = 224
    max_upload_bytes: int = 10 * 1024 * 1024
    thumb_max_px: int = 320

    prompt_mode: str = "pinch_roi"  # pinch_roi | none | hand_expanded (SAM2 only)
    pinch_close_ratio: float = 0.35
    prompt_box_expand: float = 0.35
    mask_min_area_ratio: float = 0.003
    mask_max_area_ratio: float = 0.30
    mask_max_landmark_hits: int = 6
    mask_min_pred_iou: float = 0.51

    # SAM3 localizer: a text concept plus a centred search region. The item is
    # expected to be held in the middle of the frame.
    sam3_text_prompt: str = "product"
    sam3_prompt_box: bool = True
    sam3_min_score: float = 0.30
    sam3_max_candidates: int = 8
    center_roi_width: float = 0.60
    center_roi_height: float = 0.70

    # sam2_center localizer: a fixed centre point plus the centre box. The
    # point anchors the item, so background masks cannot win. The larger area
    # cap allows an item held close to the camera.
    sam2_center_min_area_ratio: float = 0.005
    sam2_center_max_area_ratio: float = 0.60
    sam2_center_min_pred_iou: float = 0.50
    sam2_center_max_border_fraction: float = 0.08
    sam2_center_face_iou_reject: float = 0.75
    sam2_center_face_coverage_reject: float = 0.70
    sam2_center_roi_margin_ratio: float = 0.03

    @property
    def media_dir(self) -> Path:
        return self.data_dir / "media"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def hand_asset(self) -> Path:
        return self.hand_asset_path or (self.data_dir / "assets" / "hand_landmarker.task")

    @property
    def face_asset(self) -> Path:
        return self.face_asset_path or (
            self.data_dir / "assets" / "blaze_face_short_range.tflite"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
