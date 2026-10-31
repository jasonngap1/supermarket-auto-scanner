from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.dependencies import get_state
from app.errors import ImageTooLarge, InvalidImage, ModelUnavailable
from app.schemas import ScanOut, TopItemOut

router = APIRouter()


@router.post("/scan", response_model=ScanOut)
async def scan(image: UploadFile = File(...), state=Depends(get_state)):
    if state.scan is None:
        raise HTTPException(status_code=503, detail="Scan service unavailable")
    data = await image.read()
    try:
        result = await run_in_threadpool(state.scan.scan, data)
    except ImageTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidImage as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ScanOut(
        status=result.status,
        top_items=[TopItemOut(**item.__dict__) for item in result.top_items],
        timings_ms=result.timings_ms,
    )
