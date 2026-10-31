from fastapi import APIRouter, Request

from app.pipeline.variants import VARIANTS

router = APIRouter()


@router.get("/meta")
def meta(request: Request) -> dict:
    settings = request.app.state.settings
    allowed = {"v0", "v1", "v2"} | ({"v3"} if settings.enable_clip else set())
    return {
        "currency_symbol": settings.currency_symbol,
        "enable_clip": settings.enable_clip,
        "localizer": settings.localizer,
        "sam3_text_prompt": (
            settings.sam3_text_prompt if settings.localizer == "sam3" else None
        ),
        "variants": [
            {"id": v.id, "label": v.label, "crop": v.crop, "embedder": v.embedder}
            for key, v in VARIANTS.items()
            if key in allowed
        ],
    }
