from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict:
    settings = request.app.state.settings
    state = request.app.state.services
    body = {
        "status": "ok",
        "device": {"preference": settings.device},
        "assets": {"face_detector": settings.face_asset.exists()},
    }
    if state.registry is not None:
        registry_status = state.registry.status()
        body["device"].update(registry_status["device"])
        body["models"] = registry_status["models"]
    return body
