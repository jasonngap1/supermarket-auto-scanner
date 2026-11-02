"""Shared, committed catalog of enrolled items and gallery captures.

The live database lives under the gitignored ``data/`` directory. This module
mirrors its gallery state into ``catalog/`` so a fresh clone can seed itself.
"""
import json
import logging
import os
import shutil
import sqlite3
import threading
from pathlib import Path

from app.config import Settings
from app.db import db
from app.services.storage import resolve_media_path

logger = logging.getLogger(__name__)

CATALOG_FILENAME = "catalog.json"
CATALOG_VERSION = 1
_export_lock = threading.Lock()

_ITEM_FIELDS = ("id", "name", "price_cents", "created_at")
_ARTIFACT_FIELDS = ("rel_path", "thumb_rel_path", "mask_rel_path")


def catalog_root(settings: Settings) -> Path:
    return settings.catalog_dir or settings.data_dir.parent / "catalog"


def manifest_path(settings: Settings) -> Path:
    return catalog_root(settings) / CATALOG_FILENAME


def has_catalog(settings: Settings) -> bool:
    return manifest_path(settings).is_file()


def _safe_rel_path(rel_path: str) -> Path:
    path = Path(rel_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe catalog media path: {rel_path!r}")
    return path


def _current_rows(settings: Settings) -> tuple[list[dict], list[dict]]:
    with db(settings) as connection:
        items = [
            dict(row) for row in connection.execute(
                f"SELECT {', '.join(_ITEM_FIELDS)} FROM items ORDER BY id"
            ).fetchall()
        ]
        captures = [
            dict(row) for row in connection.execute(
                f"""
                SELECT id, item_id, role, session_id, rel_path, thumb_rel_path,
                       mask_rel_path, sha256, width, height,
                       0 AS hand_found, mask_found, mask_status,
                       '{{}}' AS localization_json, created_at
                FROM captures WHERE role = 'gallery' ORDER BY id
                """
            ).fetchall()
        ]
    return items, captures


def _prune_media(media_root: Path, expected: set[str]) -> None:
    if not media_root.is_dir():
        return
    for path in sorted(media_root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_file():
            rel = path.relative_to(media_root).as_posix()
            if rel not in expected:
                path.unlink(missing_ok=True)
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def export_catalog(settings: Settings) -> None:
    """Write the current gallery state and its media into the shared catalog."""
    root = catalog_root(settings)
    media_root = root / "media"
    root.mkdir(parents=True, exist_ok=True)
    media_root.mkdir(parents=True, exist_ok=True)

    items, captures = _current_rows(settings)
    expected: set[str] = set()
    for capture in captures:
        for field in _ARTIFACT_FIELDS:
            rel = capture[field]
            _safe_rel_path(rel)
            source = resolve_media_path(settings, rel)
            destination = media_root / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            expected.add(Path(rel).as_posix())

    payload = {
        "version": CATALOG_VERSION,
        "items": items,
        "captures": captures,
    }
    temporary = root / f"{CATALOG_FILENAME}.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, manifest_path(settings))
    _prune_media(media_root, expected)


def sync_catalog(settings: Settings) -> None:
    """Refresh an existing shared catalog after a local mutation.

    Best-effort: a read-only checkout must not break enrollment.
    """
    if not has_catalog(settings):
        return
    try:
        with _export_lock:
            export_catalog(settings)
    except (OSError, ValueError):
        logger.exception("Could not refresh shared catalog")


def _valid_item(item: object) -> bool:
    return (
        isinstance(item, dict)
        and isinstance(item.get("id"), int)
        and isinstance(item.get("name"), str)
        and isinstance(item.get("price_cents"), int)
        and isinstance(item.get("created_at"), str)
    )


def _valid_capture(capture: object) -> bool:
    return (
        isinstance(capture, dict)
        and isinstance(capture.get("id"), int)
        and isinstance(capture.get("item_id"), int)
        and capture.get("role", "gallery") == "gallery"
        and all(isinstance(capture.get(field), str) for field in _ARTIFACT_FIELDS)
    )


def seed_catalog(settings: Settings) -> bool:
    """Import the committed catalog into a brand-new local database."""
    path = manifest_path(settings)
    if not path.is_file():
        return False

    try:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("Invalid catalog manifest")
        if payload.get("version") != CATALOG_VERSION:
            raise ValueError("Unsupported catalog version")
        items = [item for item in payload.get("items", []) if _valid_item(item)]
        captures = [
            capture for capture in payload.get("captures", [])
            if _valid_capture(capture)
        ]
        if not items:
            return False

        with db(settings) as connection:
            existing = connection.execute("SELECT COUNT(*) FROM items").fetchone()[0]
            if existing:
                return False

            source_media = catalog_root(settings) / "media"
            if source_media.is_dir():
                shutil.copytree(source_media, settings.media_dir, dirs_exist_ok=True)

            connection.executemany(
                """
                INSERT OR IGNORE INTO items (id, name, price_cents, created_at)
                VALUES (:id, :name, :price_cents, :created_at)
                """,
                items,
            )
            connection.executemany(
                """
                INSERT OR IGNORE INTO captures (
                  id, item_id, role, session_id, rel_path, thumb_rel_path,
                  mask_rel_path, sha256, width, height, hand_found, mask_found,
                  mask_status, localization_json, created_at
                ) VALUES (
                  :id, :item_id, 'gallery', :session_id, :rel_path,
                  :thumb_rel_path, :mask_rel_path, :sha256, :width, :height,
                  :hand_found, :mask_found, :mask_status, :localization_json,
                  :created_at
                )
                """,
                [_capture_row(capture) for capture in captures],
            )
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error):
        logger.exception("Could not seed shared catalog from %s", path)
        return False
    return True


def _capture_row(capture: dict) -> dict:
    localization = capture.get("localization_json", "{}")
    if not isinstance(localization, str):
        localization = json.dumps(localization)
    return {
        "id": capture["id"],
        "item_id": capture["item_id"],
        "session_id": str(capture.get("session_id", "default")),
        "rel_path": capture["rel_path"],
        "thumb_rel_path": capture["thumb_rel_path"],
        "mask_rel_path": capture["mask_rel_path"],
        "sha256": str(capture.get("sha256", "")),
        "width": int(capture.get("width", 0)),
        "height": int(capture.get("height", 0)),
        "hand_found": int(bool(capture.get("hand_found", 0))),
        "mask_found": int(bool(capture.get("mask_found", 1))),
        "mask_status": str(capture.get("mask_status", "ok")),
        "localization_json": localization,
        "created_at": str(capture.get("created_at", "")),
    }
