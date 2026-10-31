from dataclasses import dataclass
from typing import Literal

Role = Literal["gallery", "test"]


@dataclass(frozen=True)
class Item:
    id: int
    name: str
    price_cents: int
    created_at: str
    gallery_count: int = 0
    thumb_rel_path: str | None = None


@dataclass(frozen=True)
class Capture:
    id: int
    item_id: int
    role: Role
    session_id: str
    rel_path: str
    thumb_rel_path: str
    mask_rel_path: str
    sha256: str
    width: int
    height: int
    hand_found: bool
    mask_found: bool
    mask_status: str
    localization: dict
    created_at: str


@dataclass(frozen=True)
class TopItem:
    item_id: int
    name: str
    price_cents: int
    score: float
    thumb_rel_path: str | None = None
