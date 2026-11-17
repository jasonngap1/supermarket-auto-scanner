from fastapi import APIRouter, HTTPException, Request, Response, status

from app.schemas import ItemCreate, ItemOut, ItemUpdate
from app.services.items import (
    create_item,
    delete_item,
    list_items,
    update_item,
)

router = APIRouter()


@router.get("/items", response_model=list[ItemOut])
def get_items(request: Request):
    return list_items(request.app.state.settings)


@router.post("/items", response_model=ItemOut, status_code=status.HTTP_201_CREATED)
def post_item(payload: ItemCreate, request: Request):
    return create_item(request.app.state.settings, payload.name, payload.price_cents)


@router.patch("/items/{item_id}", response_model=ItemOut)
def patch_item(item_id: int, payload: ItemUpdate, request: Request):
    data = payload.model_dump(exclude_unset=True)
    try:
        item = update_item(request.app.state.settings, item_id, **data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item


@router.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_item(item_id: int, request: Request) -> Response:
    if not delete_item(request.app.state.settings, item_id):
        raise HTTPException(status_code=404, detail="Item not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
