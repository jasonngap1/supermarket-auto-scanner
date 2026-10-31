from functools import lru_cache
from pathlib import Path

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

    sam2_model_id: str = "facebook/sam2.1-hiera-tiny"
    dinov2_model_id: str = "facebook/dinov2-small"
    face_asset_path: Path | None = None

    input_size: int = 224
    max_upload_bytes: int = 10 * 1024 * 1024
    thumb_max_px: int = 320

    # Enrollment localizer: fixed centre point plus centre box. Face, border,
    # and outside-ROI masks are rejected so gallery thumbnails stay clean.
    center_roi_width: float = 0.60
    center_roi_height: float = 0.70
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
    def face_asset(self) -> Path:
        return self.face_asset_path or (
            self.data_dir / "assets" / "blaze_face_short_range.tflite"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
