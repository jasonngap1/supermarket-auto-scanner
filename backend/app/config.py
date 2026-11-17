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
    yolo_seg_model_id: str = "yolo26n-seg.onnx"

    input_size: int = 224
    max_upload_bytes: int = 10 * 1024 * 1024
    thumb_max_px: int = 320

    # YOLO26n-seg returns the item box and mask in one pass. SAM2 is only used
    # by the v3 comparison variant. Class labels are only used to filter
    # detections, never for recognition.
    yolo_conf: float = 0.25
    yolo_iou: float = 0.45
    yolo_imgsz: int = 640
    yolo_allowed_classes: str = ""  # comma-separated COCO names or IDs; empty = all
    yolo_reject_classes: str = "person"
    yolo_min_area_ratio: float = 0.005
    yolo_max_area_ratio: float = 0.60
    yolo_center_weight: float = 0.15

    # Shared mask selection against the detection box. YOLO masks are cropped to
    # their box, so the box agreement thresholds stay looser than SAM2's.
    mask_min_box_iou: float = 0.25
    mask_min_box_coverage: float = 0.50
    mask_max_border_fraction: float = 0.10

    # SAM2's own mask quality score, used only by the v3 variant.
    sam_min_pred_iou: float = 0.50

    @property
    def media_dir(self) -> Path:
        return self.data_dir / "media"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
