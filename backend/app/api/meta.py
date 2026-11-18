from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/meta")
def meta(request: Request) -> dict:
    settings = request.app.state.settings
    return {"currency_symbol": settings.currency_symbol}
