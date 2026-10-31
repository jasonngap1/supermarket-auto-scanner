from datetime import datetime, timezone

from app.config import Settings
from app.db import db
from app.domain import Item


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_item(settings: Settings, name: str, price_cents: int) -> Item:
    name = name.strip()
    if not name:
        raise ValueError("Item name cannot be empty")
    with db(settings) as connection:
        cursor = connection.execute(
            "INSERT INTO items (name, price_cents, created_at) VALUES (?, ?, ?)",
            (name, price_cents, _now()),
        )
        item_id = int(cursor.lastrowid)
    item = get_item(settings, item_id)
    assert item is not None
    return item


def _item_query(where: str = "") -> str:
    return f"""
        SELECT i.id, i.name, i.price_cents, i.created_at,
               COALESCE(SUM(c.role = 'gallery'), 0) AS gallery_count,
               (
                 SELECT thumb.thumb_rel_path FROM captures thumb
                 WHERE thumb.item_id = i.id AND thumb.role = 'gallery'
                 ORDER BY thumb.id LIMIT 1
               ) AS thumb_rel_path
        FROM items i LEFT JOIN captures c ON c.item_id = i.id
        {where}
        GROUP BY i.id ORDER BY i.id
    """


def list_items(settings: Settings) -> list[Item]:
    with db(settings) as connection:
        rows = connection.execute(_item_query()).fetchall()
    return [Item(**dict(row)) for row in rows]


def get_item(settings: Settings, item_id: int) -> Item | None:
    with db(settings) as connection:
        row = connection.execute(
            _item_query("WHERE i.id = ?"), (item_id,)
        ).fetchone()
    return Item(**dict(row)) if row else None


def delete_item(settings: Settings, item_id: int) -> bool:
    from app.services.captures import list_captures, remove_capture_files

    captures = list_captures(settings, item_id)
    with db(settings) as connection:
        cursor = connection.execute("DELETE FROM items WHERE id = ?", (item_id,))
    if cursor.rowcount > 0:
        for capture in captures:
            remove_capture_files(settings, capture)
    return cursor.rowcount > 0
