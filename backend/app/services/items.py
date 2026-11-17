from datetime import datetime, timezone

from app.config import Settings
from app.db import db
from app.domain import Item
from app.services.catalog import sync_catalog


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
    sync_catalog(settings)
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


def update_item(
    settings: Settings,
    item_id: int,
    *,
    name: str | None = None,
    price_cents: int | None = None,
) -> Item | None:
    fields: dict[str, object] = {}
    if name is not None:
        name = name.strip()
        if not name:
            raise ValueError("Item name cannot be empty")
        fields["name"] = name
    if price_cents is not None:
        if price_cents < 0:
            raise ValueError("Price cannot be negative")
        fields["price_cents"] = int(price_cents)
    if not fields:
        raise ValueError("No fields to update")

    assignments = ", ".join(f"{column} = ?" for column in fields)
    with db(settings) as connection:
        cursor = connection.execute(
            f"UPDATE items SET {assignments} WHERE id = ?",
            (*fields.values(), item_id),
        )
    if cursor.rowcount == 0:
        return None
    return get_item(settings, item_id)


def delete_item(settings: Settings, item_id: int) -> bool:
    from app.services.captures import list_captures, remove_capture_files

    captures = list_captures(settings, item_id)
    with db(settings) as connection:
        cursor = connection.execute("DELETE FROM items WHERE id = ?", (item_id,))
    if cursor.rowcount > 0:
        for capture in captures:
            remove_capture_files(settings, capture)
        sync_catalog(settings)
    return cursor.rowcount > 0
