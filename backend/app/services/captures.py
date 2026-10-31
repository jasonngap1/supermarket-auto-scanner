import json
import sqlite3
from dataclasses import dataclass, fields
from datetime import datetime, timezone

import numpy as np

from app.config import Settings
from app.db import db
from app.domain import Capture, Role
from app.errors import DuplicateCapture, ImageTooLarge, InvalidImage, ModelUnavailable
from app.matching.vector_index import normalize
from app.pipeline.types import HandObservation, Localization, MaskSelection, PromptSpec
from app.services.storage import (
    CaptureArtifacts,
    decode_image,
    load_mask,
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
    hand_found: bool,
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
                    width, height, int(hand_found), int(mask_found), mask_status,
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


@dataclass(frozen=True)
class EvalCapture:
    id: int
    item_id: int
    item_name: str
    item_price_cents: int
    rel_path: str
    mask_rel_path: str
    mask_found: bool
    localization: dict


def list_test_captures(settings: Settings, item_ids: list[int] | None = None) -> list[EvalCapture]:
    query = """
      SELECT c.*, i.name AS item_name, i.price_cents AS item_price_cents
      FROM captures c JOIN items i ON i.id = c.item_id
      WHERE c.role = 'test'
    """
    params: list = []
    if item_ids:
        query += " AND c.item_id IN (" + ",".join("?" for _ in item_ids) + ")"
        params.extend(item_ids)
    query += " ORDER BY c.id"
    with db(settings) as connection:
        rows = connection.execute(query, params).fetchall()
    return [
        EvalCapture(
            id=row["id"], item_id=row["item_id"], item_name=row["item_name"],
            item_price_cents=row["item_price_cents"], rel_path=row["rel_path"],
            mask_rel_path=row["mask_rel_path"], mask_found=bool(row["mask_found"]),
            localization=json.loads(row["localization_json"]),
        )
        for row in rows
    ]


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
        "hand": None if localization.hand is None else {
            "landmarks_px": localization.hand.landmarks_px,
            "bbox": localization.hand.bbox,
            "score": localization.hand.score,
        },
        "prompt": None if localization.prompt is None else {
            "positive_points": localization.prompt.positive_points,
            "negative_points": localization.prompt.negative_points,
            "box": localization.prompt.box,
            "mode": localization.prompt.mode,
            "text": localization.prompt.text,
        },
        "selection": {
            "selected_index": None if selected is None else selected.index,
            "reasons": localization.selection.reasons,
            "scores": localization.selection.scores,
            "area_ratios": localization.selection.area_ratios,
        },
        "status": localization.status,
        "timings_ms": localization.timings_ms,
    }


def load_capture_localization(settings: Settings, capture) -> Localization:
    data = capture.localization
    hand_data = data.get("hand")
    hand = None
    if hand_data:
        hand = HandObservation(
            landmarks_px=[tuple(point) for point in hand_data["landmarks_px"]],
            bbox=tuple(hand_data["bbox"]),
            score=float(hand_data["score"]),
        )
    prompt_data = data.get("prompt")
    prompt = None
    if prompt_data:
        prompt = PromptSpec(
            positive_points=[tuple(p) for p in prompt_data["positive_points"]],
            negative_points=[tuple(p) for p in prompt_data["negative_points"]],
            box=None if prompt_data["box"] is None else tuple(prompt_data["box"]),
            mode=prompt_data["mode"],
            text=prompt_data.get("text"),
        )
    mask = load_mask(settings, capture.mask_rel_path) if capture.mask_found else None
    return Localization(
        hand=hand,
        prompt=prompt,
        selection=MaskSelection(None, data.get("selection", {}).get("reasons", []), [], []),
        mask=mask,
        status=data.get("status", "ok"),
        timings_ms=data.get("timings_ms", {}),
    )


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
    near_duplicate: bool = False
    near_duplicate_score: float | None = None


NEAR_DUPLICATE_THRESHOLD = 0.995


class CaptureService:
    def __init__(self, settings, pipeline, embeddings=None):
        self.settings = settings
        self.pipeline = pipeline
        self.embeddings = embeddings

    def _near_duplicate_score(self, capture: Capture, role: str) -> float | None:
        """Compare this capture against the opposite role for the same item.

        A test image that nearly duplicates a gallery image (or vice versa)
        inflates evaluation scores; surface it at enrollment time.
        """
        if self.embeddings is None:
            return None
        try:
            vector = self.embeddings.get_capture_vector(capture, "v2")
            if vector is None:
                return None
            query = normalize(vector)
            other_role = "test" if role == "gallery" else "gallery"
            best = 0.0
            for other in list_captures(self.settings, capture.item_id, other_role):
                other_vector = self.embeddings.get_capture_vector(other, "v2")
                if other_vector is not None:
                    best = max(best, float(query @ normalize(other_vector)))
            return best
        except Exception:
            return None

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
                role=role,
                session_id=session_id,
                artifacts=artifacts,
                sha256=digest,
                width=width,
                height=height,
                hand_found=localization.hand is not None,
                mask_found=localization.mask is not None,
                mask_status=localization.status,
                localization=localization_to_dict(localization),
            )
        except DuplicateCapture:
            return CaptureUploadResult(False, reason="duplicate")
        score = self._near_duplicate_score(capture, role)
        return CaptureUploadResult(
            True,
            capture=capture,
            near_duplicate=score is not None and score > NEAR_DUPLICATE_THRESHOLD,
            near_duplicate_score=score,
        )

    def list(self, item_id: int, role: str | None = None):
        return list_captures(self.settings, item_id, role)  # type: ignore[arg-type]

    def delete(self, item_id: int, capture_id: int) -> bool:
        return delete_capture(self.settings, item_id, capture_id)
