import time
import uuid
from dataclasses import dataclass, field

from app.domain import TopItem
from app.matching.matcher import item_topk
from app.pipeline.annotate import annotate_frame
from app.services.storage import decode_image

SCAN_VARIANT = "v2"


@dataclass(frozen=True)
class ScanResult:
    frame_id: str
    status: str  # ok | no_gallery | no_detection | mask_rejected | model_error
    top_items: list[TopItem] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    annotated_jpeg: bytes = b""


class ScanService:
    def __init__(self, settings, pipeline, embeddings):
        self.settings = settings
        self.pipeline = pipeline
        self.embeddings = embeddings

    def scan(self, image_bytes: bytes) -> ScanResult:
        image, _, _ = decode_image(image_bytes, self.settings.max_upload_bytes)
        entries = self.embeddings.gallery_entries(SCAN_VARIANT)
        if not entries:
            return ScanResult(
                frame_id=uuid.uuid4().hex,
                status="no_gallery",
                top_items=[],
                timings_ms={},
                annotated_jpeg=annotate_frame(
                    image, None, None, self.settings.currency_symbol
                ),
            )

        output = self.pipeline.analyze(image, [SCAN_VARIANT])
        status = output.variant_status.get(SCAN_VARIANT, "model_error")
        top_items: list[TopItem] = []
        timings = dict(output.timings_ms)
        if status == "ok":
            match_start = time.perf_counter()
            top_items = item_topk(
                {SCAN_VARIANT: output.embeddings[SCAN_VARIANT]}, entries, k=3
            )
            timings["match"] = (time.perf_counter() - match_start) * 1000
            timings["total"] = timings.get("total", 0.0) + timings["match"]

        best = top_items[0] if top_items else None
        localization = getattr(output, "localization", None)
        return ScanResult(
            frame_id=uuid.uuid4().hex,
            status=status,
            top_items=top_items,
            timings_ms=timings,
            annotated_jpeg=annotate_frame(
                image, localization, best, self.settings.currency_symbol
            ),
        )
