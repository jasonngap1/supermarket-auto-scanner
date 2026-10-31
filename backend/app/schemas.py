from pydantic import BaseModel, Field


class ItemCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    price_cents: int = Field(ge=0)


class ItemOut(BaseModel):
    id: int
    name: str
    price_cents: int
    created_at: str
    gallery_count: int = 0
    test_count: int = 0


class CaptureOut(BaseModel):
    id: int
    item_id: int
    role: str
    session_id: str
    rel_path: str
    thumb_rel_path: str
    mask_rel_path: str
    width: int
    height: int
    hand_found: bool
    mask_found: bool
    mask_status: str


class CaptureAccepted(BaseModel):
    capture: CaptureOut
    thumb_rel_path: str
    mask_rel_path: str
    mask_status: str
    near_duplicate: bool = False
    near_duplicate_score: float | None = None


class CaptureRejected(BaseModel):
    reason: str


class CaptureUploadResponse(BaseModel):
    accepted: list[CaptureAccepted]
    rejected: list[CaptureRejected]


class TopItemOut(BaseModel):
    item_id: int
    name: str
    price_cents: int
    score: float


class VariantOut(BaseModel):
    variant: str
    status: str
    top_items: list[TopItemOut]
    timings_ms: dict[str, float]
    error: str | None = None


class AnalyzeOut(BaseModel):
    frame_id: str
    annotated_jpeg_b64: str
    hand_found: bool
    mask_found: bool
    mask_status: str
    variants: list[VariantOut]


class EvaluationRequest(BaseModel):
    variants: list[str]
    item_ids: list[int] | None = None
