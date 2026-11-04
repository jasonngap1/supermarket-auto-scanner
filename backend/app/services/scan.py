import time
import uuid
from dataclasses import dataclass, field

from app.domain import TopItem
from app.matching.matcher import item_topk
from app.services.storage import decode_image


@dataclass(frozen=True)
class ScanResult:
    frame_id: str
    status: str  # ok | no_gallery
    top_items: list[TopItem] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


class ScanService:
    def __init__(self, settings, pipeline, embeddings):
        self.settings = settings
        self.pipeline = pipeline
        self.embeddings = embeddings

    def scan(self, image_bytes: bytes) -> ScanResult:
        image, _, _ = decode_image(image_bytes, self.settings.max_upload_bytes)
        entries = self.embeddings.gallery_entries()
        if not entries:
            return ScanResult(
                frame_id=uuid.uuid4().hex,
                status="no_gallery",
                top_items=[],
                timings_ms={},
            )

        vectors, timings = self.pipeline.embed_frame_variants(image)
        match_start = time.perf_counter()
        top_items = item_topk(vectors, entries, k=3)
        timings["match"] = (time.perf_counter() - match_start) * 1000
        timings["total"] = timings.get("total", 0.0) + timings["match"]

        return ScanResult(
            frame_id=uuid.uuid4().hex,
            status="ok",
            top_items=top_items,
            timings_ms=timings,
        )
