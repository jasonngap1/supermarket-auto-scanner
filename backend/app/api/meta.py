from fastapi import APIRouter, Request

from app.pipeline.variants import VARIANTS

router = APIRouter()


@router.get("/meta")
def meta(request: Request) -> dict:
    settings = request.app.state.settings
    return {
        "currency_symbol": settings.currency_symbol,
        "variants": [
            {"id": v.id, "label": v.label, "crop": v.crop}
            for v in VARIANTS.values()
        ],
    }
