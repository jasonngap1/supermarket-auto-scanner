import base64

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.dependencies import get_state
from app.errors import ImageTooLarge, InvalidImage, ModelUnavailable
from app.schemas import AnalyzeOut, TopItemOut, VariantOut

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeOut)
async def analyze(
    image: UploadFile = File(...),
    variants: str = Form("v0,v1,v2"),
    state=Depends(get_state),
):
    if state.analysis is None:
        raise HTTPException(status_code=503, detail="Analysis service unavailable")
    variant_ids = [v.strip() for v in variants.split(",") if v.strip()]
    if not variant_ids:
        raise HTTPException(status_code=422, detail="At least one variant is required")
    data = await image.read()
    try:
        result = await run_in_threadpool(state.analysis.analyze, data, variant_ids)
    except ImageTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidImage as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if result.variants and all(v.status == "model_error" for v in result.variants):
        raise HTTPException(status_code=503, detail="All requested models are unavailable")

    return AnalyzeOut(
        frame_id=result.frame_id,
        annotated_jpeg_b64=base64.b64encode(result.annotated_jpeg).decode(),
        hand_found=result.hand_found,
        mask_found=result.mask_found,
        mask_status=result.mask_status,
        variants=[
            VariantOut(
                variant=v.variant,
                status=v.status,
                top_items=[TopItemOut(**item.__dict__) for item in v.top_items],
                timings_ms=v.timings_ms,
                error=v.error,
            )
            for v in result.variants
        ],
    )
