import time
import uuid
from dataclasses import dataclass, field

from app.domain import TopItem
from app.matching.matcher import item_topk
from app.pipeline.annotate import annotate_frame
from app.services.storage import decode_image


@dataclass(frozen=True)
class VariantAnalysis:
    variant: str
    status: str
    top_items: list[TopItem] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class AnalyzeResult:
    frame_id: str
    annotated_jpeg: bytes
    hand_found: bool
    mask_found: bool
    mask_status: str
    variants: list[VariantAnalysis]


class AnalysisService:
    def __init__(self, settings, pipeline, embeddings):
        self.settings = settings
        self.pipeline = pipeline
        self.embeddings = embeddings

    def analyze(self, image_bytes: bytes, variant_ids: list[str]) -> AnalyzeResult:
        image, _, _ = decode_image(image_bytes, self.settings.max_upload_bytes)
        output = self.pipeline.analyze(image, variant_ids)
        variants: list[VariantAnalysis] = []

        for variant_id in variant_ids:
            status = output.variant_status.get(variant_id, "model_error")
            top_items: list[TopItem] = []
            match_ms = 0.0
            if status == "ok":
                match_start = time.perf_counter()
                entries = self.embeddings.gallery_entries(variant_id)
                if not entries:
                    status = "no_gallery"
                else:
                    top_items = item_topk(output.embeddings[variant_id], entries, k=3)
                match_ms = (time.perf_counter() - match_start) * 1000
            timings = {
                **{
                    key: value
                    for key, value in output.timings_ms.items()
                    if key not in ("embed", "total")
                },
                "embed": output.embed_ms.get(variant_id, 0.0),
                "match": match_ms,
            }
            timings["total"] = sum(timings.values())
            variants.append(VariantAnalysis(variant_id, status, top_items, timings))

        best = max(
            (v.top_items[0] for v in variants if v.top_items),
            key=lambda item: item.score,
            default=None,
        )
        annotated = annotate_frame(
            image, output.localization, best, self.settings.currency_symbol
        )
        return AnalyzeResult(
            frame_id=uuid.uuid4().hex,
            annotated_jpeg=annotated,
            hand_found=output.localization.hand is not None,
            mask_found=output.localization.mask is not None,
            mask_status=output.localization.status,
            variants=variants,
        )
