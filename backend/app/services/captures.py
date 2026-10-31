import json
import sqlite3
from dataclasses import dataclass, fields
from datetime import datetime, timezone

import numpy as np

from app.config import Settings
from app.db import db
from app.domain import Capture, Role
from app.errors import DuplicateCapture, ImageTooLarge, InvalidImage, ModelUnavailable
from app.pipeline.types import Localization
from app.services.storage import (
    CaptureArtifacts,
    decode_image,
    resolve_media_path,
    save_capture_bundle,
    sha256_hex,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_to_capture(row: sqlite3.Row) -> Capture:
    data = dict(row)
    data["hand_found"] = bool(data["hand_found"])
    data["mask_found"] = bool(data["mask_found"])
    data["localization"] = json.loads(data.pop("localization_json"))
    allowed = {field.name for field in fields(Capture)}
    return Capture(**{key: value for key, value in data.items() if key in allowed})


def add_capture(
    settings: Settings,
    *,
    item_id: int,
    role: Role,
    session_id: str,
    artifacts: CaptureArtifacts,
    sha256: str,
    width: int,
    height: int,
    mask_found: bool,
    mask_status: str,
    localization: dict,
) -> Capture:
    try:
        with db(settings) as connection:
            cursor = connection.execute(
                """
                INSERT INTO captures (
                  item_id, role, session_id, rel_path, thumb_rel_path, mask_rel_path,
                  sha256, width, height, hand_found, mask_found, mask_status,
                  localization_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id, role, session_id, artifacts.rel_path,
                    artifacts.thumb_rel_path, artifacts.mask_rel_path, sha256,
                    width, height, 0, int(mask_found), mask_status,
                    json.dumps(localization), _now(),
                ),
            )
            capture_id = int(cursor.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise DuplicateCapture("This image is already enrolled for this item/role") from exc
    capture = get_capture(settings, capture_id)
    assert capture is not None
    return capture


def get_capture(settings: Settings, capture_id: int) -> Capture | None:
    with db(settings) as connection:
        row = connection.execute("SELECT * FROM captures WHERE id = ?", (capture_id,)).fetchone()
    return _row_to_capture(row) if row else None


def list_captures(
    settings: Settings, item_id: int, role: Role | None = None
) -> list[Capture]:
    query = "SELECT * FROM captures WHERE item_id = ?"
    params: list = [item_id]
    if role:
        query += " AND role = ?"
        params.append(role)
    query += " ORDER BY id"
    with db(settings) as connection:
        rows = connection.execute(query, params).fetchall()
    return [_row_to_capture(row) for row in rows]


def list_gallery_captures(settings: Settings):
    """Return (capture, item_name, price_cents) for all gallery captures."""
    with db(settings) as connection:
        rows = connection.execute(
            """
            SELECT c.*, i.name, i.price_cents FROM captures c
            JOIN items i ON i.id = c.item_id
            WHERE c.role = 'gallery' ORDER BY c.item_id, c.id
            """
        ).fetchall()
    return [(_row_to_capture(row), row["name"], row["price_cents"]) for row in rows]


def remove_capture_files(settings: Settings, capture: Capture) -> None:
    for rel_path in (capture.rel_path, capture.thumb_rel_path, capture.mask_rel_path):
        try:
            resolve_media_path(settings, rel_path).unlink(missing_ok=True)
        except (OSError, ValueError):
            pass


def delete_capture(settings: Settings, item_id: int, capture_id: int) -> bool:
    capture = get_capture(settings, capture_id)
    if capture is None or capture.item_id != item_id:
        return False
    with db(settings) as connection:
        cursor = connection.execute(
            "DELETE FROM captures WHERE id = ? AND item_id = ?", (capture_id, item_id)
        )
    if cursor.rowcount > 0:
        remove_capture_files(settings, capture)
    return cursor.rowcount > 0


def upsert_embedding(
    settings: Settings,
    capture_id: int,
    variant: str,
    model_key: str,
    preprocess_version: str,
    vector: np.ndarray,
) -> None:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    with db(settings) as connection:
        connection.execute(
            """
            INSERT INTO embeddings
              (capture_id, variant, model_key, preprocess_version, dim, vector, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(capture_id, variant, model_key, preprocess_version)
            DO UPDATE SET vector = excluded.vector, dim = excluded.dim
            """,
            (
                capture_id, variant, model_key, preprocess_version, vector.size,
                vector.tobytes(), _now(),
            ),
        )


def get_embedding(
    settings: Settings,
    capture_id: int,
    variant: str,
    model_key: str,
    preprocess_version: str,
) -> np.ndarray | None:
    with db(settings) as connection:
        row = connection.execute(
            """
            SELECT vector FROM embeddings
            WHERE capture_id = ? AND variant = ? AND model_key = ? AND preprocess_version = ?
            """,
            (capture_id, variant, model_key, preprocess_version),
        ).fetchone()
    if not row:
        return None
    return np.frombuffer(row["vector"], dtype=np.float32).copy()


def localization_to_dict(localization: Localization) -> dict:
    selected = localization.selection.selected
    return {
        "prompt": None if localization.prompt is None else {
            "positive_points": localization.prompt.positive_points,
            "negative_points": localization.prompt.negative_points,
            "box": localization.prompt.box,
            "mode": localization.prompt.mode,
        },
        "selection": {
            "selected_index": None if selected is None else selected.index,
            "reasons": localization.selection.reasons,
            "scores": localization.selection.scores,
            "area_ratios": localization.selection.area_ratios,
        },
        "face": None if localization.face is None else {
            "bbox": localization.face.bbox,
            "score": localization.face.score,
        },
        "status": localization.status,
        "timings_ms": localization.timings_ms,
    }


def capture_duplicate_exists(settings: Settings, item_id: int, role: str, sha256: str) -> bool:
    with db(settings) as connection:
        row = connection.execute(
            "SELECT 1 FROM captures WHERE item_id = ? AND role = ? AND sha256 = ?",
            (item_id, role, sha256),
        ).fetchone()
    return row is not None


@dataclass(frozen=True)
class CaptureUploadResult:
    accepted: bool
    capture: Capture | None = None
    reason: str | None = None


class CaptureService:
    def __init__(self, settings, pipeline):
        self.settings = settings
        self.pipeline = pipeline

    def add(self, item_id: int, role: str, session_id: str, data: bytes) -> CaptureUploadResult:
        try:
            image, width, height = decode_image(data, self.settings.max_upload_bytes)
        except (InvalidImage, ImageTooLarge) as exc:
            return CaptureUploadResult(False, reason=str(exc))

        digest = sha256_hex(data)
        if capture_duplicate_exists(self.settings, item_id, role, digest):
            return CaptureUploadResult(False, reason="duplicate")

        try:
            localization = self.pipeline.localize(image)
        except ModelUnavailable:
            return CaptureUploadResult(False, reason="model_error")
        if localization.status != "ok" or localization.mask is None:
            return CaptureUploadResult(False, reason=localization.status)

        artifacts = save_capture_bundle(self.settings, item_id, image, localization.mask)
        try:
            capture = add_capture(
                self.settings,
                item_id=item_id,
                role=role,  # type: ignore[arg-type]
                session_id=session_id,
                artifacts=artifacts,
                sha256=digest,
                width=width,
                height=height,
                mask_found=localization.mask is not None,
                mask_status=localization.status,
                localization=localization_to_dict(localization),
            )
        except DuplicateCapture:
            return CaptureUploadResult(False, reason="duplicate")
        return CaptureUploadResult(True, capture=capture)

    def list(self, item_id: int, role: str | None = None):
        return list_captures(self.settings, item_id, role)  # type: ignore[arg-type]

    def delete(self, item_id: int, capture_id: int) -> bool:
        return delete_capture(self.settings, item_id, capture_id)
