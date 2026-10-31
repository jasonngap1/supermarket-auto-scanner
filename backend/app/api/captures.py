from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from app.dependencies import get_state
from app.schemas import (
    CaptureAccepted,
    CaptureOut,
    CaptureRejected,
    CaptureUploadResponse,
)
from app.services.items import get_item

router = APIRouter()


def _capture_out(capture) -> CaptureOut:
    return CaptureOut(
        id=capture.id, item_id=capture.item_id, role=capture.role,
        session_id=capture.session_id, rel_path=capture.rel_path,
        thumb_rel_path=capture.thumb_rel_path, mask_rel_path=capture.mask_rel_path,
        width=capture.width, height=capture.height, hand_found=capture.hand_found,
        mask_found=capture.mask_found, mask_status=capture.mask_status,
    )


@router.get("/items/{item_id}/captures", response_model=list[CaptureOut])
def get_captures(item_id: int, role: str | None = None, state=Depends(get_state)):
    if get_item(state.settings, item_id) is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return [_capture_out(c) for c in state.captures.list(item_id, role)]


@router.post(
    "/items/{item_id}/captures",
    response_model=CaptureUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_captures(
    item_id: int,
    files: list[UploadFile] = File(...),
    role: str = "gallery",
    session_id: str = "default",
    state=Depends(get_state),
):
    if role not in ("gallery", "test"):
        raise HTTPException(status_code=422, detail="role must be gallery or test")
    if get_item(state.settings, item_id) is None:
        raise HTTPException(status_code=404, detail="Item not found")
    if len(files) > 10:
        raise HTTPException(status_code=422, detail="At most 10 files per request")
    accepted, rejected = [], []
    for upload in files:
        data = await upload.read()
        result = await run_in_threadpool(
            state.captures.add, item_id, role, session_id, data
        )
        if result.accepted and result.capture is not None:
            accepted.append(
                CaptureAccepted(
                    capture=_capture_out(result.capture),
                    thumb_rel_path=result.capture.thumb_rel_path,
                    mask_rel_path=result.capture.mask_rel_path,
                    mask_status=result.capture.mask_status,
                    near_duplicate=result.near_duplicate,
                    near_duplicate_score=result.near_duplicate_score,
                )
            )
        else:
            rejected.append(CaptureRejected(reason=result.reason or "rejected"))
    return CaptureUploadResponse(accepted=accepted, rejected=rejected)


@router.delete("/items/{item_id}/captures/{capture_id}", status_code=204)
def delete_capture_route(item_id: int, capture_id: int, state=Depends(get_state)):
    if not state.captures.delete(item_id, capture_id):
        raise HTTPException(status_code=404, detail="Capture not found")
    return Response(status_code=204)
