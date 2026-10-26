# Supermarket Auto-Scanner Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local FastAPI + Vite model-validation lab that recognizes a supermarket item held up to the laptop camera, retrieves its name and price, and measures whether hand-aware SAM2 segmentation + DINOv2 embedding is accurate enough for a final scanner app.

**Architecture:** One FastAPI process owns all inference (MediaPipe hand detection → SAM2 mask → DINOv2/CLIP embedding → exact cosine match) behind a single inference gate; SQLite and local files store items, captures, masks, embeddings, and evaluation runs. A Vite vanilla-JS UI provides three tabs: Scan (~1 fps live), Enroll (webcam gallery/test captures), and Evaluate (batch metrics).

**Tech Stack:** Python 3.12 via `uv`, FastAPI, SQLite, PyTorch, Transformers (SAM2, DINOv2, CLIP), MediaPipe Tasks HandLandmarker, NumPy, Pillow; Vite vanilla JavaScript, Vitest; pytest.

## Global Constraints

- Python **3.12 only**, managed by `uv` (`backend/.python-version`). Never use system Python 3.14.
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` **before importing torch**; `app/bootstrap.py` is the first app import in `main.py`.
- One uvicorn worker; all model calls go through one global `threading.RLock` inference gate.
- Float32 inference only; no autocast/bf16 on MPS.
- Backend binds `127.0.0.1`; no auth, no remote access, no cloud uploads.
- Price is stored as integer **cents**; display uses `AUTO_SCANNER_CURRENCY_SYMBOL` (default `$`).
- Uploads capped at **10 MB**, UUID filenames, paths validated under `data/media`.
- Evaluation is **batch/offline** over labeled test captures. Scan is live inference only and reports no accuracy.
- SAM2 produces masks only. Retrieval vectors come only from DINOv2 (primary) or CLIP (baseline); no fused embeddings, no SAM2 encoder features.
- Exact cosine matching only; HNSW is deferred behind a `VectorIndex` interface.
- Fast tests must not download weights. Model tests are `@pytest.mark.slow`, run only with `RUN_MODEL_TESTS=1`.
- CV wrappers must not import FastAPI; API modules must not import torch/mediapipe at module import time.
- Commit after every task. Do not commit `data/`, model weights, or `node_modules`.

## File Map

```
backend/
  .python-version                        3.12
  pyproject.toml  uv.lock
  app/
    __init__.py
    bootstrap.py                         env vars before torch
    config.py                            Settings, paths, thresholds
    main.py                              create_app(), lifespan, router wiring, /media mount
    db.py  schema.sql                    SQLite connection + full schema
    domain.py                            Item, Capture, TopItem, analysis dataclasses
    schemas.py                           Pydantic request/response models
    errors.py                            ImageTooLarge, InvalidImage, DuplicateCapture, ModelUnavailable
    dependencies.py                      AppState + get_state()
    api/__init__.py
    api/health.py  meta.py  items.py  captures.py  analyze.py  evaluations.py
    services/items.py                    item CRUD
    services/storage.py                  image/mask/thumb files, path safety
    services/captures.py                 capture CRUD + upload service
    services/embeddings.py               lazy per-capture embedding cache
    services/analysis.py                 one-frame orchestration + matching
    services/evaluation.py               background evaluation orchestration
    pipeline/types.py                    HandObservation, PromptSpec, MaskCandidate, Localization
    pipeline/prompts.py                  landmark → point/box prompts
    pipeline/mask_select.py              candidate filtering/ranking
    pipeline/preprocess.py               crop + letterbox
    pipeline/hand.py                     MediaPipe wrapper
    pipeline/sam2.py                     SAM2 wrapper
    pipeline/embedders.py                DINOv2 / CLIP wrappers
    pipeline/variants.py                 V0–V3 definitions
    pipeline/frame_pipeline.py           hand → segment → embed orchestration
    pipeline/annotate.py                 overlay + JPEG rendering
    ml/devices.py                        MPS/CPU resolution
    ml/registry.py                       lazy models, gate, device fallback, health
    matching/vector_index.py             ExactCosineIndex + protocol
    matching/matcher.py                  item-level top-k
    evaluation/metrics.py                accuracy, Wilson CI, per-item, confusion, latency
    evaluation/runner.py                 evaluation loop
    evaluation/artifacts.py              failure frame rendering/saving
  assets/                                hand_landmarker.task (downloaded, gitignored)
  tests/
    conftest.py fakes.py
    test_health.py test_items_api.py test_storage.py test_prompts.py
    test_mask_select.py test_preprocess.py test_vector_index.py test_metrics.py
    test_frame_pipeline.py test_captures_api.py test_analyze_api.py
    test_evaluations_api.py test_repository.py test_models_slow.py
frontend/
  package.json  package-lock.json  vite.config.js  index.html
  src/main.js api.js camera.js format.js format.test.js style.css
  src/tabs/scan.js enroll.js evaluate.js
scripts/
  download_models.py smoke_models.py run_backend.sh run_frontend.sh
data/                                    gitignored
docs/superpowers/specs/2026-10-26-supermarket-scanner-design.md
docs/superpowers/plans/2026-10-26-supermarket-scanner.md
```

---

### Task 1: Backend skeleton, settings, health endpoint

**Files:**
- Create: `backend/.python-version`, `backend/pyproject.toml`, `backend/app/__init__.py`, `backend/app/bootstrap.py`, `backend/app/config.py`, `backend/app/main.py`, `backend/app/api/__init__.py`, `backend/app/api/health.py`
- Test: `backend/tests/test_health.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Settings` (fields below), `get_settings()`, `create_app(settings: Settings | None = None) -> FastAPI`; `GET /api/health` returns `{"status": "ok", "device": {"preference": str}, "assets": {"hand_landmarker": bool}}`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_health.py
import importlib
import os

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_bootstrap_sets_mps_fallback_before_torch(monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    import app.bootstrap

    importlib.reload(app.bootstrap)
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_health_endpoint(tmp_path):
    settings = Settings(data_dir=tmp_path, warm_models=False)
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["device"]["preference"] == "auto"
    assert body["assets"]["hand_landmarker"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv sync --python 3.12 && uv run pytest tests/test_health.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app'`.

- [ ] **Step 3: Write minimal implementation**

```toml
# backend/pyproject.toml
[project]
name = "supermarket-auto-scanner-backend"
version = "0.1.0"
description = "Local held-item recognition model validation lab"
requires-python = ">=3.12,<3.13"
dependencies = [
  "fastapi>=0.115",
  "uvicorn>=0.30",
  "python-multipart>=0.0.9",
  "pydantic-settings>=2.4",
  "pillow>=10.4",
  "numpy>=1.26",
  "torch>=2.6",
  "torchvision>=0.21",
  "transformers>=4.52,<6",
  "mediapipe>=1.0,<2",
  "huggingface-hub>=0.26",
]

[dependency-groups]
dev = ["pytest>=8", "pytest-timeout>=2.3", "httpx>=0.27"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: requires downloaded model weights and RUN_MODEL_TESTS=1"]
```

```text
# backend/.python-version
3.12
```

```python
# backend/app/__init__.py
```

```python
# backend/app/bootstrap.py
"""Must be imported before torch. Sets MPS fallback and cache locations."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("AUTO_SCANNER_DATA_DIR", str(PROJECT_ROOT / "data")))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", str(DATA_DIR / "hf_cache"))
```

```python
# backend/app/config.py
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AUTO_SCANNER_", env_file=".env", extra="ignore"
    )

    data_dir: Path = PROJECT_ROOT / "data"
    device: str = "auto"  # auto | mps | cpu
    currency_symbol: str = "$"
    warm_models: bool = True
    enable_clip: bool = False

    sam2_model_id: str = "facebook/sam2.1-hiera-tiny"
    dinov2_model_id: str = "facebook/dinov2-small"
    clip_model_id: str = "openai/clip-vit-base-patch32"
    hand_asset_path: Path | None = None

    input_size: int = 224
    max_upload_bytes: int = 10 * 1024 * 1024
    thumb_max_px: int = 320

    prompt_mode: str = "pinch_roi"  # pinch_roi | none | hand_expanded
    pinch_close_ratio: float = 0.35
    prompt_box_expand: float = 0.35
    mask_min_area_ratio: float = 0.003
    mask_max_area_ratio: float = 0.30
    mask_max_landmark_hits: int = 6

    @property
    def media_dir(self) -> Path:
        return self.data_dir / "media"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def hand_asset(self) -> Path:
        return self.hand_asset_path or (self.data_dir / "assets" / "hand_landmarker.task")


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

```python
# backend/app/api/__init__.py
```

```python
# backend/app/api/health.py
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict:
    settings = request.app.state.settings
    return {
        "status": "ok",
        "device": {"preference": settings.device},
        "assets": {"hand_landmarker": settings.hand_asset.exists()},
    }
```

```python
# backend/app/main.py
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import bootstrap  # noqa: F401  (must run before torch)
from app.api.health import router as health_router
from app.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.media_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield

    app = FastAPI(title="Supermarket Auto-Scanner", lifespan=lifespan)
    app.state.settings = settings
    app.include_router(health_router, prefix="/api")
    return app


app = create_app()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv sync --python 3.12 && uv run pytest tests/test_health.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/.python-version backend/pyproject.toml backend/uv.lock backend/app backend/tests/test_health.py
git commit -m "feat: backend skeleton with settings and health endpoint"
```

---

### Task 2: SQLite schema, item repository, items API

**Files:**
- Create: `backend/app/schema.sql`, `backend/app/db.py`, `backend/app/domain.py`, `backend/app/schemas.py`, `backend/app/services/__init__.py`, `backend/app/services/items.py`, `backend/app/api/items.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_items_api.py`

**Interfaces:**
- Consumes: `Settings` from Task 1.
- Produces: `db(settings)` context manager, `init_db(settings)`; `create_item(settings, name, price_cents) -> Item`, `list_items(settings) -> list[Item]`, `get_item(settings, item_id) -> Item | None`, `delete_item(settings, item_id) -> bool`; `GET/POST/DELETE /api/items`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_items_api.py
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def make_client(tmp_path):
    return TestClient(create_app(Settings(data_dir=tmp_path, warm_models=False)))


def test_create_list_delete_item_and_persist(tmp_path):
    with make_client(tmp_path) as client:
        created = client.post("/api/items", json={"name": "Milk 1L", "price_cents": 199})
        assert created.status_code == 201
        item = created.json()
        assert item["name"] == "Milk 1L"
        assert item["price_cents"] == 199
        assert item["gallery_count"] == 0

        listed = client.get("/api/items").json()
        assert [i["id"] for i in listed] == [item["id"]]

    # New app instance, same data dir: data persists.
    with make_client(tmp_path) as client:
        assert client.get("/api/items").json()[0]["name"] == "Milk 1L"

    with make_client(tmp_path) as client:
        deleted = client.delete(f"/api/items/{item['id']}")
        assert deleted.status_code == 204
        assert client.get("/api/items").json() == []


def test_rejects_negative_price(tmp_path):
    with make_client(tmp_path) as client:
        response = client.post("/api/items", json={"name": "Bad", "price_cents": -1})
    assert response.status_code == 422
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_items_api.py -v`
Expected: FAIL because `/api/items` does not exist.

- [ ] **Step 3: Write minimal implementation**

```sql
-- backend/app/schema.sql
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
  version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS items (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  name        TEXT NOT NULL,
  price_cents INTEGER NOT NULL CHECK (price_cents >= 0),
  created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS captures (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  item_id           INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  role              TEXT NOT NULL CHECK (role IN ('gallery', 'test')),
  session_id        TEXT NOT NULL,
  rel_path          TEXT NOT NULL UNIQUE,
  thumb_rel_path    TEXT NOT NULL,
  mask_rel_path     TEXT NOT NULL,
  sha256            TEXT NOT NULL,
  width             INTEGER NOT NULL,
  height            INTEGER NOT NULL,
  hand_found        INTEGER NOT NULL DEFAULT 0,
  mask_found        INTEGER NOT NULL DEFAULT 0,
  mask_status       TEXT NOT NULL,
  localization_json TEXT NOT NULL,
  created_at        TEXT NOT NULL,
  UNIQUE (item_id, role, sha256)
);
CREATE INDEX IF NOT EXISTS idx_captures_item_role ON captures(item_id, role);

CREATE TABLE IF NOT EXISTS embeddings (
  capture_id         INTEGER NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
  variant            TEXT NOT NULL,
  model_key          TEXT NOT NULL,
  preprocess_version TEXT NOT NULL,
  dim                INTEGER NOT NULL,
  vector             BLOB NOT NULL,
  created_at         TEXT NOT NULL,
  PRIMARY KEY (capture_id, variant, model_key, preprocess_version)
);
CREATE INDEX IF NOT EXISTS idx_embeddings_lookup
  ON embeddings(variant, model_key, preprocess_version);

CREATE TABLE IF NOT EXISTS eval_runs (
  id             TEXT PRIMARY KEY,
  created_at     TEXT NOT NULL,
  started_at     TEXT,
  finished_at    TEXT,
  status         TEXT NOT NULL CHECK (status IN ('queued','running','completed','failed')),
  request_json   TEXT NOT NULL,
  progress_done  INTEGER NOT NULL DEFAULT 0,
  progress_total INTEGER NOT NULL DEFAULT 0,
  metrics_json   TEXT,
  error          TEXT
);

CREATE TABLE IF NOT EXISTS eval_samples (
  id                 INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id             TEXT NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
  capture_id         INTEGER,
  variant            TEXT NOT NULL,
  true_item_id       INTEGER,
  true_item_name     TEXT NOT NULL,
  true_price_cents   INTEGER NOT NULL,
  pred_item_id       INTEGER,
  pred_item_name     TEXT,
  topk_json          TEXT NOT NULL,
  correct_top1       INTEGER NOT NULL DEFAULT 0,
  correct_top5       INTEGER NOT NULL DEFAULT 0,
  sample_status      TEXT NOT NULL,
  error              TEXT,
  timings_json       TEXT NOT NULL,
  rendered_rel_path  TEXT,
  created_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_samples_run_variant
  ON eval_samples(run_id, variant, correct_top1);
```

```python
# backend/app/db.py
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from app.config import Settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(settings: Settings) -> sqlite3.Connection:
    connection = sqlite3.connect(settings.db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


@contextmanager
def db(settings: Settings):
    connection = connect(settings)
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db(settings: Settings) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.media_dir.mkdir(parents=True, exist_ok=True)
    with db(settings) as connection:
        connection.executescript(SCHEMA_PATH.read_text())
```

```python
# backend/app/domain.py
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
    test_count: int = 0


@dataclass(frozen=True)
class TopItem:
    item_id: int
    name: str
    price_cents: int
    score: float
```

```python
# backend/app/schemas.py
from pydantic import BaseModel, Field


class ItemCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    price_cents: int = Field(ge=0)


class ItemOut(BaseModel):
    id: int
    name: str
    price_cents: int
    created_at: str
    gallery_count: int = 0
    test_count: int = 0
```

```python
# backend/app/services/__init__.py
```

```python
# backend/app/services/items.py
from datetime import datetime, timezone

from app.config import Settings
from app.db import db
from app.domain import Item


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_item(settings: Settings, name: str, price_cents: int) -> Item:
    with db(settings) as connection:
        cursor = connection.execute(
            "INSERT INTO items (name, price_cents, created_at) VALUES (?, ?, ?)",
            (name.strip(), price_cents, _now()),
        )
        item_id = int(cursor.lastrowid)
    return get_item(settings, item_id)  # type: ignore[return-value]


def list_items(settings: Settings) -> list[Item]:
    with db(settings) as connection:
        rows = connection.execute(
            """
            SELECT i.id, i.name, i.price_cents, i.created_at,
                   COALESCE(SUM(c.role = 'gallery'), 0) AS gallery_count,
                   COALESCE(SUM(c.role = 'test'), 0) AS test_count
            FROM items i LEFT JOIN captures c ON c.item_id = i.id
            GROUP BY i.id ORDER BY i.id
            """
        ).fetchall()
    return [Item(**dict(row)) for row in rows]


def get_item(settings: Settings, item_id: int) -> Item | None:
    with db(settings) as connection:
        row = connection.execute(
            """
            SELECT i.id, i.name, i.price_cents, i.created_at,
                   COALESCE(SUM(c.role = 'gallery'), 0) AS gallery_count,
                   COALESCE(SUM(c.role = 'test'), 0) AS test_count
            FROM items i LEFT JOIN captures c ON c.item_id = i.id
            WHERE i.id = ? GROUP BY i.id
            """,
            (item_id,),
        ).fetchone()
    return Item(**dict(row)) if row else None


def delete_item(settings: Settings, item_id: int) -> bool:
    with db(settings) as connection:
        cursor = connection.execute("DELETE FROM items WHERE id = ?", (item_id,))
    return cursor.rowcount > 0
```

```python
# backend/app/api/items.py
from fastapi import APIRouter, HTTPException, Request, Response, status

from app.schemas import ItemCreate, ItemOut
from app.services.items import create_item, delete_item, list_items

router = APIRouter()


@router.get("/items", response_model=list[ItemOut])
def get_items(request: Request):
    return list_items(request.app.state.settings)


@router.post("/items", response_model=ItemOut, status_code=status.HTTP_201_CREATED)
def post_item(payload: ItemCreate, request: Request):
    return create_item(request.app.state.settings, payload.name, payload.price_cents)


@router.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_item(item_id: int, request: Request) -> Response:
    if not delete_item(request.app.state.settings, item_id):
        raise HTTPException(status_code=404, detail="Item not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
```

Modify `backend/app/main.py` lifespan and router wiring:

```python
from app.api.items import router as items_router
from app.db import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(settings)
    yield

# inside create_app after health router:
    app.include_router(items_router, prefix="/api")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_items_api.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/schema.sql backend/app/db.py backend/app/domain.py backend/app/schemas.py backend/app/services backend/app/api/items.py backend/app/main.py backend/tests/test_items_api.py
git commit -m "feat: sqlite schema and item CRUD API"
```

---

### Task 3: File storage, capture store, embedding cache table access

**Files:**
- Create: `backend/app/errors.py`, `backend/app/pipeline/__init__.py`, `backend/app/pipeline/types.py`, `backend/app/services/storage.py`, `backend/app/services/captures.py`
- Modify: `backend/app/domain.py`
- Test: `backend/tests/test_storage.py`, `backend/tests/test_repository.py`

**Interfaces:**
- Consumes: `Settings`, `db`, schema from Task 2.
- Produces: `decode_image(data, max_bytes) -> (np.ndarray, width, height)`; `sha256_hex(data) -> str`; `save_capture_bundle(settings, item_id, image_rgb, mask) -> CaptureArtifacts`; `resolve_media_path(settings, rel_path) -> Path`; `add_capture(...) -> Capture`; `list_captures(settings, item_id, role=None) -> list[Capture]`; `get_capture`, `delete_capture`; `upsert_embedding`, `get_embedding`, `list_gallery_vectors`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_storage.py
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from app.config import Settings
from app.errors import DuplicateCapture, InvalidImage, ImageTooLarge
from app.services.storage import (
    decode_image,
    resolve_media_path,
    save_capture_bundle,
    sha256_hex,
)


def jpeg_bytes(width=64, height=48, color=(200, 30, 30)) -> bytes:
    image = Image.new("RGB", (width, height), color)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_decode_rejects_bad_and_large_payloads(tmp_path):
    with pytest.raises(InvalidImage):
        decode_image(b"not an image", max_bytes=1000)
    with pytest.raises(ImageTooLarge):
        decode_image(jpeg_bytes(), max_bytes=10)


def test_save_capture_bundle_and_path_safety(tmp_path):
    settings = Settings(data_dir=tmp_path)
    image, width, height = decode_image(jpeg_bytes(), max_bytes=10_000)
    mask = np.zeros((height, width), dtype=bool)
    mask[10:40, 10:50] = True
    artifacts = save_capture_bundle(settings, item_id=7, image_rgb=image, mask=mask)
    for rel_path in (artifacts.rel_path, artifacts.thumb_rel_path, artifacts.mask_rel_path):
        path = resolve_media_path(settings, rel_path)
        assert path.exists()
        assert settings.media_dir in path.parents
    assert (width, height) == (64, 48)
    assert len(sha256_hex(jpeg_bytes())) == 64


def test_rejects_path_traversal(tmp_path):
    settings = Settings(data_dir=tmp_path)
    with pytest.raises(ValueError):
        resolve_media_path(settings, "../app.db")
```

```python
# backend/tests/test_repository.py
import numpy as np
import pytest

from app.config import Settings
from app.errors import DuplicateCapture
from app.services.captures import (
    add_capture,
    delete_item_captures,
    get_embedding,
    list_captures,
    upsert_embedding,
)
from app.services.items import create_item
from app.services.storage import save_capture_bundle
from app.db import init_db


def setup_item(tmp_path):
    settings = Settings(data_dir=tmp_path)
    init_db(settings)
    item = create_item(settings, "Milk", 199)
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    mask = np.ones((32, 32), dtype=bool)
    artifacts = save_capture_bundle(settings, item.id, image, mask)
    return settings, item, artifacts


def make_capture(settings, item, artifacts, role="gallery", sha="a" * 64):
    return add_capture(
        settings,
        item_id=item.id,
        role=role,
        session_id="s1",
        artifacts=artifacts,
        sha256=sha,
        width=32,
        height=32,
        hand_found=True,
        mask_found=True,
        mask_status="ok",
        localization={"hand": {}, "selection": {}},
    )


def test_capture_duplicate_and_cascade_delete(tmp_path):
    settings, item, artifacts = setup_item(tmp_path)
    capture = make_capture(settings, item, artifacts)
    assert list_captures(settings, item.id)[0].id == capture.id
    with pytest.raises(DuplicateCapture):
        make_capture(settings, item, artifacts)
    delete_item_captures(settings, item.id)
    assert list_captures(settings, item.id) == []


def test_embedding_round_trip(tmp_path):
    settings, item, artifacts = setup_item(tmp_path)
    capture = make_capture(settings, item, artifacts)
    vector = np.linspace(0, 1, 384, dtype=np.float32)
    upsert_embedding(settings, capture.id, "v2", "dinov2-small", "prep-v1", vector)
    loaded = get_embedding(settings, capture.id, "v2", "dinov2-small", "prep-v1")
    assert loaded is not None
    assert np.allclose(loaded, vector)
    assert get_embedding(settings, capture.id, "v3", "clip", "prep-v1") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/test_storage.py tests/test_repository.py -v`
Expected: FAIL with `ModuleNotFoundError: app.errors` / missing functions.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/app/domain.py (append)
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
```

```python
# backend/app/errors.py
class InvalidImage(ValueError):
    pass


class ImageTooLarge(ValueError):
    pass


class DuplicateCapture(ValueError):
    pass


class ModelUnavailable(RuntimeError):
    pass


class CaptureRejected(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason
```

```python
# backend/app/pipeline/__init__.py
```

```python
# backend/app/pipeline/types.py
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class HandObservation:
    landmarks_px: list[tuple[float, float]]
    bbox: tuple[float, float, float, float]
    score: float


@dataclass(frozen=True)
class PromptSpec:
    positive_points: list[tuple[float, float]]
    negative_points: list[tuple[float, float]]
    box: tuple[float, float, float, float] | None
    mode: str


@dataclass(frozen=True)
class MaskCandidate:
    index: int
    mask: np.ndarray
    pred_iou: float


@dataclass(frozen=True)
class MaskSelection:
    selected: MaskCandidate | None
    reasons: list[str]
    scores: list[float]
    area_ratios: list[float]


@dataclass(frozen=True)
class Localization:
    hand: HandObservation | None
    prompt: PromptSpec | None
    selection: MaskSelection
    mask: np.ndarray | None
    status: str  # ok | no_hand | mask_rejected | model_error
    timings_ms: dict[str, float] = field(default_factory=dict)
```

```python
# backend/app/services/storage.py
import hashlib
import math
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.config import Settings
from app.errors import ImageTooLarge, InvalidImage


@dataclass(frozen=True)
class CaptureArtifacts:
    rel_path: str
    thumb_rel_path: str
    mask_rel_path: str


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode_image(data: bytes, max_bytes: int) -> tuple[np.ndarray, int, int]:
    if len(data) > max_bytes:
        raise ImageTooLarge(f"Upload exceeds {max_bytes} bytes")
    try:
        image = Image.open(BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImage("Unsupported or corrupt image") from exc
    image = image.convert("RGB")
    return np.asarray(image, dtype=np.uint8), image.width, image.height


def resolve_media_path(settings: Settings, rel_path: str) -> Path:
    candidate = (settings.media_dir / rel_path).resolve()
    if settings.media_dir.resolve() not in candidate.parents:
        raise ValueError("Path escapes media directory")
    return candidate


def save_capture_bundle(
    settings: Settings, item_id: int, image_rgb: np.ndarray, mask: np.ndarray
) -> CaptureArtifacts:
    stem = uuid.uuid4().hex
    image_rel = f"images/{item_id}/{stem}.jpg"
    mask_rel = f"masks/{stem}.png"
    thumb_rel = f"thumbs/{stem}.jpg"

    image_path = resolve_media_path(settings, image_rel)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb).save(image_path, format="JPEG", quality=90)

    mask_path = resolve_media_path(settings, mask_rel)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_path, format="PNG")

    overlay = image_rgb.astype(np.float32).copy()
    overlay[mask] = overlay[mask] * 0.55 + np.array([0, 120, 255]) * 0.45
    thumb = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
    thumb.thumbnail((settings.thumb_max_px, settings.thumb_max_px))
    thumb_path = resolve_media_path(settings, thumb_rel)
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    thumb.save(thumb_path, format="JPEG", quality=85)

    return CaptureArtifacts(image_rel, thumb_rel, mask_rel)


def load_image_rgb(settings: Settings, rel_path: str) -> np.ndarray:
    with Image.open(resolve_media_path(settings, rel_path)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_mask(settings: Settings, rel_path: str) -> np.ndarray:
    with Image.open(resolve_media_path(settings, rel_path)) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 127
```

```python
# backend/app/services/captures.py
import json
import sqlite3
from dataclasses import fields
from datetime import datetime, timezone

import numpy as np

from app.config import Settings
from app.db import db
from app.domain import Capture, Role
from app.errors import DuplicateCapture
from app.services.storage import CaptureArtifacts


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
                    item_id, role, session_id, artifacts.rel_path, artifacts.thumb_rel_path,
                    artifacts.mask_rel_path, sha256, width, height, int(hand_found),
                    int(mask_found), mask_status, json.dumps(localization), _now(),
                ),
            )
            capture_id = int(cursor.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise DuplicateCapture("This image is already enrolled for this item/role") from exc
    return get_capture(settings, capture_id)  # type: ignore[return-value]


def _row_to_capture(row: sqlite3.Row) -> Capture:
    data = dict(row)
    data["hand_found"] = bool(data["hand_found"])
    data["mask_found"] = bool(data["mask_found"])
    data["localization"] = json.loads(data.pop("localization_json"))
    allowed = {field.name for field in fields(Capture)}
    return Capture(**{key: value for key, value in data.items() if key in allowed})


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


def delete_capture(settings: Settings, item_id: int, capture_id: int) -> bool:
    with db(settings) as connection:
        cursor = connection.execute(
            "DELETE FROM captures WHERE id = ? AND item_id = ?", (capture_id, item_id)
        )
    return cursor.rowcount > 0


def delete_item_captures(settings: Settings, item_id: int) -> None:
    with db(settings) as connection:
        connection.execute("DELETE FROM captures WHERE item_id = ?", (item_id,))


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
            SELECT vector, dim FROM embeddings
            WHERE capture_id = ? AND variant = ? AND model_key = ? AND preprocess_version = ?
            """,
            (capture_id, variant, model_key, preprocess_version),
        ).fetchone()
    if not row:
        return None
    return np.frombuffer(row["vector"], dtype=np.float32).copy()


def list_gallery_vectors(
    settings: Settings, variant: str, model_key: str, preprocess_version: str
) -> list[tuple[int, int, str, int, np.ndarray]]:
    """Return (capture_id, item_id, item_name, price_cents, vector) for gallery captures."""
    with db(settings) as connection:
        rows = connection.execute(
            """
            SELECT e.capture_id, c.item_id, i.name, i.price_cents, e.vector
            FROM embeddings e
            JOIN captures c ON c.id = e.capture_id
            JOIN items i ON i.id = c.item_id
            WHERE c.role = 'gallery' AND e.variant = ? AND e.model_key = ?
              AND e.preprocess_version = ?
            ORDER BY c.id
            """,
            (variant, model_key, preprocess_version),
        ).fetchall()
    return [
        (
            row["capture_id"], row["item_id"], row["name"], row["price_cents"],
            np.frombuffer(row["vector"], dtype=np.float32).copy(),
        )
        for row in rows
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_storage.py tests/test_repository.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/errors.py backend/app/pipeline/__init__.py backend/app/pipeline/types.py backend/app/services/storage.py backend/app/services/captures.py backend/tests/test_storage.py backend/tests/test_repository.py
git commit -m "feat: file storage, capture repository, embedding cache"
```

---

### Task 4: Hand-landmark prompt geometry

**Files:**
- Create: `backend/app/pipeline/prompts.py`
- Test: `backend/tests/test_prompts.py`

**Interfaces:**
- Consumes: `HandObservation`, `PromptSpec` from Task 3.
- Produces: `build_prompt(landmarks_px, frame_w, frame_h, *, mode="pinch_roi", pinch_close_ratio=0.35, box_expand=0.35) -> PromptSpec`; `hand_crop_box(landmarks_px, frame_w, frame_h, expand=0.35) -> tuple`; `hand_bbox`, `hand_scale`, `clip_box`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_prompts.py
from app.pipeline.prompts import build_prompt, hand_crop_box


def landmarks(thumb=(40, 40), index=(60, 40), middle=(65, 50)):
    points = [(50.0, 90.0)] * 21
    points[0] = (50.0, 90.0)   # wrist
    points[4] = thumb          # thumb tip
    points[5] = (45.0, 70.0)
    points[8] = index          # index tip
    points[9] = (55.0, 70.0)
    points[12] = middle        # middle tip
    points[13] = (60.0, 72.0)
    points[17] = (65.0, 75.0)
    return points


def test_open_pinch_uses_pinch_center_and_excludes_fingertips_from_negatives():
    prompt = build_prompt(landmarks(), 200, 200)
    assert prompt.positive_points == [(50.0, 40.0)]
    assert (40.0, 40.0) not in prompt.negative_points
    assert (60.0, 40.0) not in prompt.negative_points
    assert (50.0, 90.0) in prompt.negative_points  # wrist
    assert prompt.box is not None
    x1, y1, x2, y2 = prompt.box
    assert x1 <= 50 <= x2 and y1 <= 40 <= y2


def test_closed_pinch_uses_grasp_center():
    prompt = build_prompt(landmarks(thumb=(50, 40), index=(52, 40)), 200, 200)
    assert prompt.positive_points == [(54.25, 42.5)]


def test_box_mode_none_and_clipping():
    prompt = build_prompt(landmarks(thumb=(2, 2), index=(4, 2)), 100, 100, mode="none")
    assert prompt.box is None
    expanded = build_prompt(landmarks(thumb=(2, 2), index=(4, 2)), 100, 100)
    assert expanded.box[0] >= 0 and expanded.box[1] >= 0
    crop = hand_crop_box(landmarks(thumb=(2, 2), index=(4, 2)), 100, 100)
    assert crop[0] >= 0 and crop[1] >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError: app.pipeline.prompts`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/app/pipeline/prompts.py
import math

from app.pipeline.types import PromptSpec

NEGATIVE_INDICES = (0, 5, 9, 13, 17)
GRASP_INDICES = (4, 8, 12)


def _mid(a, b):
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _mean(points):
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def _clip_point(point, width, height):
    return (
        min(max(point[0], 0.0), float(width - 1)),
        min(max(point[1], 0.0), float(height - 1)),
    )


def clip_box(box, width, height):
    x1, y1, x2, y2 = box
    x1, y1 = _clip_point((x1, y1), width, height)
    x2, y2 = _clip_point((x2, y2), width, height)
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def hand_bbox(landmarks):
    xs = [p[0] for p in landmarks]
    ys = [p[1] for p in landmarks]
    return (min(xs), min(ys), max(xs), max(ys))


def hand_scale(landmarks):
    x1, y1, x2, y2 = hand_bbox(landmarks)
    return max(x2 - x1, y2 - y1)


def _expand_box(box, margin, width, height):
    x1, y1, x2, y2 = box
    return clip_box((x1 - margin, y1 - margin, x2 + margin, y2 + margin), width, height)


def hand_crop_box(landmarks_px, frame_w, frame_h, expand=0.35):
    box = hand_bbox(landmarks_px)
    margin = expand * max(box[2] - box[0], box[3] - box[1])
    return _expand_box(box, margin, frame_w, frame_h)


def build_prompt(
    landmarks_px,
    frame_w,
    frame_h,
    *,
    mode="pinch_roi",
    pinch_close_ratio=0.35,
    box_expand=0.35,
) -> PromptSpec:
    if len(landmarks_px) != 21:
        raise ValueError("Expected 21 hand landmarks")

    landmarks = [_clip_point(p, frame_w, frame_h) for p in landmarks_px]
    palm_center = _mean([landmarks[i] for i in NEGATIVE_INDICES])
    pinch_center = _mid(landmarks[4], landmarks[8])
    grasp_center = _mid(landmarks[4], _mid(landmarks[8], landmarks[12]))
    scale = hand_scale(landmarks)
    pinch_distance = math.dist(landmarks[4], landmarks[8])
    positive = grasp_center if pinch_distance < pinch_close_ratio * scale else pinch_center
    negative = [landmarks[i] for i in NEGATIVE_INDICES] + [palm_center]
    # Keep unique negatives, preserving order.
    negative = list(dict.fromkeys(negative))

    if mode == "none":
        box = None
    elif mode == "hand_expanded":
        box = hand_crop_box(landmarks, frame_w, frame_h, box_expand)
    elif mode == "pinch_roi":
        roi = hand_bbox([landmarks[4], landmarks[8], landmarks[12], palm_center])
        box = _expand_box(roi, box_expand * scale, frame_w, frame_h)
    else:
        raise ValueError(f"Unknown prompt mode: {mode}")

    return PromptSpec(
        positive_points=[positive],
        negative_points=negative,
        box=box,
        mode=mode,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_prompts.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/prompts.py backend/tests/test_prompts.py
git commit -m "feat: hand landmark prompt geometry"
```

---

### Task 5: SAM2 mask-candidate selection

**Files:**
- Create: `backend/app/pipeline/mask_select.py`
- Test: `backend/tests/test_mask_select.py`

**Interfaces:**
- Consumes: `MaskCandidate`, `MaskSelection` from Task 3.
- Produces: `select_mask(candidates, landmarks_px, positive, negatives, grasp_roi, frame_area, *, min_area_ratio, max_area_ratio, max_landmark_hits) -> MaskSelection`; `mask_bbox`, `point_inside`, `bbox_overlaps`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_mask_select.py
import numpy as np

from app.pipeline.mask_select import mask_bbox, select_mask
from app.pipeline.types import MaskCandidate


def make_mask(x1, y1, x2, y2, size=40):
    mask = np.zeros((size, size), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def run(candidates, landmarks=None, positive=(20, 20), negatives=((5, 5),), roi=(0, 0, 40, 40)):
    return select_mask(
        candidates,
        landmarks or [(5, 5)] * 21,
        positive,
        list(negatives),
        roi,
        frame_area=40 * 40,
        min_area_ratio=0.003,
        max_area_ratio=0.30,
        max_landmark_hits=6,
    )


def test_selects_mask_containing_positive_and_avoiding_negatives():
    good = MaskCandidate(0, make_mask(10, 10, 30, 30), 0.9)
    hand = MaskCandidate(1, make_mask(0, 0, 40, 40), 0.99)
    result = run([hand, good], negatives=((5, 5),))
    assert result.selected is not None
    assert result.selected.index == 0
    assert "negative_inside" in result.reasons[0]


def test_rejects_positive_outside_and_too_small():
    outside = MaskCandidate(0, make_mask(0, 0, 10, 10), 0.9)
    tiny = MaskCandidate(1, make_mask(20, 20, 22, 22), 0.95)
    result = run([outside, tiny], positive=(21, 21))
    assert result.selected is None
    assert result.reasons[0] == "positive_outside"
    assert result.reasons[1] == "area_too_small"


def test_rejects_too_many_landmarks_and_no_roi_overlap():
    big = MaskCandidate(0, make_mask(0, 0, 30, 30), 0.9)
    far = MaskCandidate(1, make_mask(30, 30, 38, 38), 0.9)
    landmarks = [(5, 5)] * 7 + [(20, 20)] * 14
    result = run(
        [big, far],
        landmarks=landmarks,
        positive=(20, 20),
        negatives=((39, 39),),
        roi=(10, 10, 30, 30),
    )
    assert result.selected is None
    assert result.reasons[0] == "too_many_landmarks"
    assert result.reasons[1] == "positive_outside"


def test_bbox_helper():
    assert mask_bbox(make_mask(3, 4, 8, 9)) == (3, 4, 8, 9)
    assert mask_bbox(np.zeros((5, 5), dtype=bool)) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_mask_select.py -v`
Expected: FAIL with `ModuleNotFoundError: app.pipeline.mask_select`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/app/pipeline/mask_select.py
import numpy as np

from app.pipeline.types import MaskCandidate, MaskSelection


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def point_inside(mask: np.ndarray, point: tuple[float, float], pad: int = 2) -> bool:
    height, width = mask.shape[:2]
    x = int(round(point[0]))
    y = int(round(point[1]))
    x1, x2 = max(0, x - pad), min(width, x + pad + 1)
    y1, y2 = max(0, y - pad), min(height, y + pad + 1)
    if x1 >= x2 or y1 >= y2:
        return False
    return bool(mask[y1:y2, x1:x2].any())


def landmark_hits(mask: np.ndarray, landmarks_px) -> int:
    return sum(1 for point in landmarks_px if point_inside(mask, point, pad=0))


def bbox_overlaps(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def select_mask(
    candidates: list[MaskCandidate],
    landmarks_px,
    positive_point,
    negative_points,
    grasp_roi,
    frame_area: int,
    *,
    min_area_ratio: float,
    max_area_ratio: float,
    max_landmark_hits: int,
) -> MaskSelection:
    reasons: list[str] = []
    scores: list[float] = []
    areas: list[float] = []

    for candidate in candidates:
        mask = candidate.mask
        box = mask_bbox(mask)
        area_ratio = float(mask.sum()) / float(frame_area)
        hits = landmark_hits(mask, landmarks_px)
        areas.append(area_ratio)

        if box is None:
            reason = "empty"
        elif not point_inside(mask, positive_point):
            reason = "positive_outside"
        elif any(point_inside(mask, point) for point in negative_points):
            reason = "negative_inside"
        elif hits > max_landmark_hits:
            reason = "too_many_landmarks"
        elif area_ratio < min_area_ratio:
            reason = "area_too_small"
        elif area_ratio > max_area_ratio:
            reason = "area_too_large"
        elif grasp_roi is not None and not bbox_overlaps(box, grasp_roi):
            reason = "no_roi_overlap"
        else:
            reason = "ok"

        area_penalty = 0.05 * max(0.0, area_ratio - 0.15) / 0.15
        score = float(candidate.pred_iou) - 0.05 * hits - area_penalty
        reasons.append(reason)
        scores.append(score)

    eligible = [
        (scores[i], -areas[i], candidates[i])
        for i, reason in enumerate(reasons)
        if reason == "ok"
    ]
    selected = max(eligible, key=lambda item: (item[0], item[1]))[2] if eligible else None
    return MaskSelection(selected=selected, reasons=reasons, scores=scores, area_ratios=areas)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_mask_select.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/mask_select.py backend/tests/test_mask_select.py
git commit -m "feat: sam2 mask candidate selection"
```

---

### Task 6: Crop and letterbox preprocessing

**Files:**
- Create: `backend/app/pipeline/preprocess.py`
- Test: `backend/tests/test_preprocess.py`

**Interfaces:**
- Consumes: `mask_bbox` from Task 5.
- Produces: `PREPROCESS_VERSION = "letterbox224-gray10-v1"`; `crop_box(image, box, margin_ratio=0.0)`; `crop_mask(image, mask, margin_ratio=0.10, background=127)`; `letterbox(image, size=224, fill=127)`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_preprocess.py
import numpy as np

from app.pipeline.preprocess import (
    PREPROCESS_VERSION,
    crop_box,
    crop_mask,
    letterbox,
)


def test_letterbox_preserves_aspect_and_pads():
    image = np.full((60, 120, 3), 10, dtype=np.uint8)
    out = letterbox(image, size=224)
    assert out.shape == (224, 224, 3)
    assert tuple(out[0, 0]) == (127, 127, 127)
    assert tuple(out[100, 100]) == (10, 10, 10)


def test_crop_mask_grays_outside_mask():
    image = np.full((40, 40, 3), 200, dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    mask[15:25, 15:25] = False  # hole in the middle
    out = crop_mask(image, mask, margin_ratio=0.0)
    assert out.shape == (20, 20, 3)
    assert tuple(out[0, 0]) == (200, 200, 200)
    assert tuple(out[10, 10]) == (127, 127, 127)


def test_crop_box_clips_and_margins():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    crop = crop_box(image, (-5, -5, 10, 10), margin_ratio=0.0)
    assert crop.shape[:2] == (10, 10)
    assert PREPROCESS_VERSION == "letterbox224-gray10-v1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_preprocess.py -v`
Expected: FAIL with `ModuleNotFoundError: app.pipeline.preprocess`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/app/pipeline/preprocess.py
import math

import numpy as np
from PIL import Image

from app.pipeline.mask_select import mask_bbox

PREPROCESS_VERSION = "letterbox224-gray10-v1"


def crop_box(image: np.ndarray, box, margin_ratio: float = 0.0) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    margin_x = (x2 - x1) * margin_ratio
    margin_y = (y2 - y1) * margin_ratio
    x1 = max(0, int(math.floor(x1 - margin_x)))
    y1 = max(0, int(math.floor(y1 - margin_y)))
    x2 = min(width, int(math.ceil(x2 + margin_x)))
    y2 = min(height, int(math.ceil(y2 + margin_y)))
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Empty crop box")
    return image[y1:y2, x1:x2]


def crop_mask(
    image: np.ndarray, mask: np.ndarray, margin_ratio: float = 0.10, background: int = 127
) -> np.ndarray:
    box = mask_bbox(mask)
    if box is None:
        raise ValueError("Empty mask")
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    margin_x = (x2 - x1) * margin_ratio
    margin_y = (y2 - y1) * margin_ratio
    x1 = max(0, int(math.floor(x1 - margin_x)))
    y1 = max(0, int(math.floor(y1 - margin_y)))
    x2 = min(width, int(math.ceil(x2 + margin_x)))
    y2 = min(height, int(math.ceil(y2 + margin_y)))
    crop = image[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]
    out = np.full_like(crop, background)
    out[mask_crop] = crop[mask_crop]
    return out


def letterbox(image: np.ndarray, size: int = 224, fill: int = 127) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))
    resized = Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (fill, fill, fill))
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return np.asarray(canvas, dtype=np.uint8)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_preprocess.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/preprocess.py backend/tests/test_preprocess.py
git commit -m "feat: mask crop and letterbox preprocessing"
```

---

### Task 7: Exact cosine index and item-level matcher

**Files:**
- Create: `backend/app/matching/__init__.py`, `backend/app/matching/vector_index.py`, `backend/app/matching/matcher.py`
- Test: `backend/tests/test_vector_index.py`

**Interfaces:**
- Consumes: `TopItem` from `app.domain`.
- Produces: `VectorIndex` Protocol; `ExactCosineIndex.add(ids, vectors)`, `.search(query, k) -> list[tuple[int, float]]`; `GalleryEntry(item_id, name, price_cents, vectors)`; `item_topk(query, entries, k=3) -> list[TopItem]`; `normalize(vector)`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_vector_index.py
import numpy as np
import pytest

from app.matching.matcher import GalleryEntry, item_topk
from app.matching.vector_index import ExactCosineIndex, normalize


def test_exact_index_ranks_by_cosine():
    index = ExactCosineIndex()
    index.add([10, 20], np.array([[1, 0], [0, 1]], dtype=np.float32))
    assert index.search(np.array([0.9, 0.1]), k=2) == [(10, pytest.approx(0.9938837)), (20, pytest.approx(0.11043153))]


def test_normalize_is_safe_for_zero_vector():
    assert np.allclose(normalize(np.zeros(3, dtype=np.float32)), 0)


def test_item_topk_takes_max_over_gallery_angles():
    entries = [
        GalleryEntry(1, "Milk", 199, np.array([[1, 0], [0.8, 0.6]], dtype=np.float32)),
        GalleryEntry(2, "Bread", 250, np.array([[0, 1]], dtype=np.float32)),
    ]
    top = item_topk(np.array([1, 0], dtype=np.float32), entries, k=3)
    assert [t.item_id for t in top] == [1, 2]
    assert top[0].score == pytest.approx(1.0)
    assert len(item_topk(np.array([1, 0], dtype=np.float32), entries, k=5)) == 2


def test_empty_gallery_returns_empty():
    assert item_topk(np.array([1, 0], dtype=np.float32), [], k=3) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_vector_index.py -v`
Expected: FAIL with `ModuleNotFoundError: app.matching`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/app/matching/__init__.py
```

```python
# backend/app/matching/vector_index.py
from typing import Protocol

import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


class VectorIndex(Protocol):
    def add(self, ids: list[int], vectors: np.ndarray) -> None: ...
    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]: ...


class ExactCosineIndex:
    """Exact cosine search; replace with an HNSW implementation when N grows."""

    def __init__(self) -> None:
        self._ids: list[int] = []
        self._matrix = np.zeros((0, 0), dtype=np.float32)

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        self._ids.extend(ids)
        self._matrix = matrix if self._matrix.size == 0 else np.vstack([self._matrix, matrix])

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        if not self._ids:
            return []
        scores = self._matrix @ normalize(query)
        order = np.argsort(-scores)[:k]
        return [(self._ids[int(i)], float(scores[int(i)])) for i in order]
```

```python
# backend/app/matching/matcher.py
from dataclasses import dataclass

import numpy as np

from app.domain import TopItem
from app.matching.vector_index import normalize


@dataclass(frozen=True)
class GalleryEntry:
    item_id: int
    name: str
    price_cents: int
    vectors: np.ndarray  # shape (num_angles, dim), L2-normalized


def item_topk(query: np.ndarray, entries: list[GalleryEntry], k: int = 3) -> list[TopItem]:
    query = normalize(query)
    scored: list[TopItem] = []
    for entry in entries:
        vectors = np.asarray(entry.vectors, dtype=np.float32)
        if vectors.size == 0:
            continue
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        scores = (vectors / norms) @ query
        scored.append(
            TopItem(
                item_id=entry.item_id,
                name=entry.name,
                price_cents=entry.price_cents,
                score=float(np.max(scores)),
            )
        )
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:k]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_vector_index.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/matching backend/tests/test_vector_index.py
git commit -m "feat: exact cosine index and item-level matcher"
```

---

### Task 8: Evaluation metrics

**Files:**
- Create: `backend/app/evaluation/__init__.py`, `backend/app/evaluation/metrics.py`
- Test: `backend/tests/test_metrics.py`

**Interfaces:**
- Consumes: `TopItem` from Task 2.
- Produces: `SampleResult`; `wilson_ci(successes, n, z=1.96) -> (low, high)`; `percentile(values, p)`; `compute_metrics(samples, n_items) -> dict` with keys `n`, `n_items`, `top1`, `top5`, `macro_top1`, `top1_ci`, `k_effective`, `hand_found_rate`, `mask_accepted_rate`, `per_item`, `confusion`, `margin_p10`, `margin_p50`, `latency_ms`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_metrics.py
import pytest

from app.domain import TopItem
from app.evaluation.metrics import SampleResult, compute_metrics, wilson_ci


def sample(true_id, top_ids, status="ok", scores=None):
    scores = scores or [0.9 - 0.1 * i for i in range(len(top_ids))]
    return SampleResult(
        true_item_id=true_id,
        true_item_name=f"item-{true_id}",
        true_price_cents=100,
        top=[TopItem(i, f"item-{i}", 100, s) for i, s in zip(top_ids, scores)],
        status=status,
        timings_ms={"embed": 10.0, "match": 1.0, "total": 11.0},
    )


def test_wilson_interval_edges():
    assert wilson_ci(0, 10)[0] == pytest.approx(0.0)
    low, high = wilson_ci(10, 10)
    assert low < 1.0 and high == pytest.approx(1.0)


def test_compute_metrics_micro_macro_and_failures():
    samples = [
        sample(1, [1, 2]),
        sample(1, [2, 1]),
        sample(2, [3, 2]),
        sample(3, [], status="no_hand"),
    ]
    metrics = compute_metrics(samples, n_items=3)
    assert metrics["n"] == 4
    assert metrics["top1"] == pytest.approx(0.25)
    assert metrics["top5"] == pytest.approx(0.5)
    assert metrics["macro_top1"] == pytest.approx((0.5 + 0.0 + 0.0) / 3)
    assert metrics["k_effective"] == 3
    assert metrics["n_items"] == 3
    assert metrics["hand_found_rate"] == pytest.approx(0.75)
    assert metrics["mask_accepted_rate"] == pytest.approx(0.75)
    assert metrics["per_item"]["item-1"]["top1"] == pytest.approx(0.5)
    assert metrics["confusion"]["item-2"]["item-3"] == 1
    assert metrics["latency_ms"]["total"]["p50"] == pytest.approx(11.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: app.evaluation`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/app/evaluation/__init__.py
```

```python
# backend/app/evaluation/metrics.py
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from app.domain import TopItem


@dataclass(frozen=True)
class SampleResult:
    true_item_id: int
    true_item_name: str
    true_price_cents: int
    top: list[TopItem]
    status: str
    timings_ms: dict[str, float]


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else 0.0


def compute_metrics(samples: list[SampleResult], n_items: int) -> dict:
    n = len(samples)
    correct1 = [bool(s.top and s.top[0].item_id == s.true_item_id) for s in samples]
    correct5 = [
        bool(any(item.item_id == s.true_item_id for item in s.top[:5])) for s in samples
    ]

    per_item: dict[str, dict] = defaultdict(lambda: {"n": 0, "correct_top1": 0, "correct_top5": 0})
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    margins: list[float] = []

    for sample, ok1, ok5 in zip(samples, correct1, correct5):
        bucket = per_item[sample.true_item_name]
        bucket["n"] += 1
        bucket["correct_top1"] += int(ok1)
        bucket["correct_top5"] += int(ok5)
        if sample.top:
            confusion[sample.true_item_name][sample.top[0].name] += 1
        true_score = next(
            (item.score for item in sample.top if item.item_id == sample.true_item_id), 0.0
        )
        best_other = max(
            (item.score for item in sample.top if item.item_id != sample.true_item_id),
            default=0.0,
        )
        margins.append(true_score - best_other)

    per_item_out = {
        name: {
            "n": bucket["n"],
            "top1": bucket["correct_top1"] / bucket["n"],
            "top5": bucket["correct_top5"] / bucket["n"],
        }
        for name, bucket in per_item.items()
    }
    macro = (
        sum(values["top1"] for values in per_item_out.values()) / len(per_item_out)
        if per_item_out
        else 0.0
    )
    total1 = sum(correct1)
    stages = sorted({key for s in samples for key in s.timings_ms})
    latency = {
        stage: {
            "p50": percentile([s.timings_ms.get(stage, 0.0) for s in samples], 50),
            "p95": percentile([s.timings_ms.get(stage, 0.0) for s in samples], 95),
        }
        for stage in stages
    }
    hand_found_rate = (
        sum(1 for sample in samples if sample.status != "no_hand") / n if n else 0.0
    )
    mask_accepted_rate = (
        sum(
            1 for sample in samples
            if sample.status not in {"no_hand", "mask_rejected"}
        ) / n if n else 0.0
    )

    return {
        "n": n,
        "n_items": n_items,
        "top1": total1 / n if n else 0.0,
        "top5": sum(correct5) / n if n else 0.0,
        "macro_top1": macro,
        "top1_ci": wilson_ci(total1, n),
        "k_effective": min(5, n_items),
        "hand_found_rate": hand_found_rate,
        "mask_accepted_rate": mask_accepted_rate,
        "per_item": per_item_out,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "margin_p10": percentile(margins, 10),
        "margin_p50": percentile(margins, 50),
        "latency_ms": latency,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_metrics.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/evaluation backend/tests/test_metrics.py
git commit -m "feat: evaluation metrics with wilson intervals"
```

### Task 9: MediaPipe hand detector wrapper

**Files:**
- Create: `backend/app/pipeline/hand.py`
- Test: `backend/tests/test_hand.py`

**Interfaces:**
- Consumes: `HandObservation` from Task 3; `ModelUnavailable` from Task 3.
- Produces: `HandDetector(asset_path, num_hands=1, min_confidence=0.5)` with `.detect(image_rgb) -> HandObservation | None`; pure helper `observation_from_landmarks(landmarks, width, height, score=1.0) -> HandObservation`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_hand.py
from types import SimpleNamespace

import numpy as np
import pytest

from app.errors import ModelUnavailable
from app.pipeline.hand import HandDetector, observation_from_landmarks


def test_observation_from_landmarks_converts_normalized_coords():
    landmarks = [SimpleNamespace(x=0.25, y=0.50) for _ in range(21)]
    observation = observation_from_landmarks(landmarks, width=400, height=200, score=0.9)
    assert observation.landmarks_px[0] == (100.0, 100.0)
    assert observation.bbox == (100.0, 100.0, 100.0, 100.0)
    assert observation.score == 0.9


def test_missing_asset_is_a_clean_model_error(tmp_path):
    detector = HandDetector(asset_path=tmp_path / "missing.task")
    with pytest.raises(ModelUnavailable):
        detector.detect(np.zeros((32, 32, 3), dtype=np.uint8))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_hand.py -v`
Expected: FAIL with `ModuleNotFoundError: app.pipeline.hand`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/app/pipeline/hand.py
from pathlib import Path

import numpy as np

from app.errors import ModelUnavailable
from app.pipeline.types import HandObservation


def observation_from_landmarks(landmarks, width: int, height: int, score: float = 1.0):
    points = [(float(lm.x) * width, float(lm.y) * height) for lm in landmarks]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return HandObservation(
        landmarks_px=points,
        bbox=(min(xs), min(ys), max(xs), max(ys)),
        score=float(score),
    )


class HandDetector:
    def __init__(self, asset_path: Path, num_hands: int = 1, min_confidence: float = 0.5):
        if not Path(asset_path).exists():
            raise ModelUnavailable(f"Hand landmarker asset not found: {asset_path}")
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(asset_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=num_hands,
                min_hand_detection_confidence=min_confidence,
                min_hand_presence_confidence=min_confidence,
                min_tracking_confidence=min_confidence,
            )
        )

    def detect(self, image_rgb: np.ndarray) -> HandObservation | None:
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(image_rgb),
        )
        result = self._landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return None
        score = 1.0
        if result.handedness and result.handedness[0]:
            score = float(result.handedness[0][0].score)
        height, width = image_rgb.shape[:2]
        return observation_from_landmarks(result.hand_landmarks[0], width, height, score)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_hand.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/hand.py backend/tests/test_hand.py
git commit -m "feat: mediapipe hand detector wrapper"
```

---

### Task 10: DINOv2 and CLIP embedder wrappers

**Files:**
- Create: `backend/app/pipeline/embedders.py`
- Test: `backend/tests/test_embedders.py`

**Interfaces:**
- Consumes: `normalize` from Task 7; `ModelUnavailable` from Task 3.
- Produces: `DinoV2Embedder(model_id, device)` and `ClipEmbedder(model_id, device)`, each with `.model_key`, `.dim`, `.embed(image_rgb) -> np.ndarray` (L2-normalized float32). Models load eagerly in `__init__`; the registry controls lazy construction.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_embedders.py
from app.pipeline.embedders import ClipEmbedder, DinoV2Embedder


def test_embedder_contract():
    dino = object.__new__(DinoV2Embedder)
    dino.model_key, dino.dim = "dinov2-small", 384
    clip = object.__new__(ClipEmbedder)
    clip.model_key, clip.dim = "clip-vit-b32", 512
    assert dino.model_key == "dinov2-small" and dino.dim == 384
    assert clip.model_key == "clip-vit-b32" and clip.dim == 512
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_embedders.py -v`
Expected: FAIL with `ModuleNotFoundError: app.pipeline.embedders`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/app/pipeline/embedders.py
import numpy as np
from PIL import Image

from app.matching.vector_index import normalize


class DinoV2Embedder:
    model_key = "dinov2-small"
    dim = 384

    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.device = device
        self._torch = torch
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id).to(device).eval()

    def embed(self, image_rgb: np.ndarray) -> np.ndarray:
        inputs = self._processor(images=Image.fromarray(image_rgb), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
        vector = outputs.last_hidden_state[:, 0].squeeze(0).float().cpu().numpy()
        return normalize(vector)


class ClipEmbedder:
    model_key = "clip-vit-b32"
    dim = 512

    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self._torch = torch
        self._processor = CLIPProcessor.from_pretrained(model_id)
        self._model = CLIPModel.from_pretrained(model_id).to(device).eval()

    def embed(self, image_rgb: np.ndarray) -> np.ndarray:
        inputs = self._processor(images=Image.fromarray(image_rgb), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            features = self._model.get_image_features(**inputs)
        return normalize(features.squeeze(0).float().cpu().numpy())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_embedders.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/embedders.py backend/tests/test_embedders.py
git commit -m "feat: dinov2 and clip embedder wrappers"
```

---

### Task 11: SAM2 segmenter wrapper

**Files:**
- Create: `backend/app/pipeline/sam2.py`
- Test: `backend/tests/test_sam2.py`

**Interfaces:**
- Consumes: `PromptSpec`, `MaskCandidate` from Task 3; `ModelUnavailable` from Task 3.
- Produces: `Sam2Segmenter(model_id, device)` with `.segment(image_rgb, prompt) -> list[MaskCandidate]`; pure helper `prompt_payload(prompt) -> dict` used by tests.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_sam2.py
from app.pipeline.sam2 import prompt_payload
from app.pipeline.types import PromptSpec


def test_prompt_payload_has_positive_and_negative_labels():
    prompt = PromptSpec(
        positive_points=[(10.0, 20.0)],
        negative_points=[(1.0, 2.0), (3.0, 4.0)],
        box=(0.0, 0.0, 30.0, 40.0),
        mode="pinch_roi",
    )
    payload = prompt_payload(prompt)
    assert payload["input_points"] == [[[[10.0, 20.0], [1.0, 2.0], [3.0, 4.0]]]]
    assert payload["input_labels"] == [[[1, 0, 0]]]
    assert payload["input_boxes"] == [[[0.0, 0.0, 30.0, 40.0]]]


def test_prompt_payload_omits_box_when_none():
    prompt = PromptSpec([(1.0, 1.0)], [], None, "none")
    assert "input_boxes" not in prompt_payload(prompt)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_sam2.py -v`
Expected: FAIL with `ModuleNotFoundError: app.pipeline.sam2`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/app/pipeline/sam2.py
import numpy as np
from PIL import Image

from app.pipeline.types import MaskCandidate, PromptSpec


def prompt_payload(prompt: PromptSpec) -> dict:
    points = [*prompt.positive_points, *prompt.negative_points]
    labels = [1] * len(prompt.positive_points) + [0] * len(prompt.negative_points)
    payload = {
        "input_points": [[[list(map(float, point)) for point in points]]],
        "input_labels": [[labels]],
    }
    if prompt.box is not None:
        payload["input_boxes"] = [[list(map(float, prompt.box))]]
    return payload


class Sam2Segmenter:
    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import Sam2Model, Sam2Processor

        self.device = device
        self._torch = torch
        self._processor = Sam2Processor.from_pretrained(model_id)
        self._model = Sam2Model.from_pretrained(model_id).to(device).eval()

    def segment(self, image_rgb: np.ndarray, prompt: PromptSpec) -> list[MaskCandidate]:
        payload = prompt_payload(prompt)
        payload.update(images=Image.fromarray(image_rgb), return_tensors="pt")
        inputs = self._processor(**payload).to(self.device)
        with self._torch.inference_mode():
            outputs = self._model(**inputs, multimask_output=True)

        # SAM2 versions differ slightly in post_process_masks arguments; support both.
        try:
            masks = self._processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"],
                inputs.get("reshaped_input_sizes"),
            )[0]
        except TypeError:
            masks = self._processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"]
            )[0]

        mask_array = np.asarray(masks)
        if mask_array.ndim == 4:
            mask_array = mask_array[0]
        if mask_array.ndim == 2:
            mask_array = mask_array[None, ...]

        iou_array = np.asarray(outputs.iou_scores[0].detach().float().cpu()).reshape(-1)
        count = min(mask_array.shape[0], iou_array.size)
        return [
            MaskCandidate(index=i, mask=mask_array[i] > 0, pred_iou=float(iou_array[i]))
            for i in range(count)
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_sam2.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/sam2.py backend/tests/test_sam2.py
git commit -m "feat: sam2 segmenter wrapper"
```

---

### Task 12: Device selection and model registry

**Files:**
- Create: `backend/app/ml/__init__.py`, `backend/app/ml/devices.py`, `backend/app/ml/registry.py`, `backend/app/dependencies.py`
- Modify: `backend/app/main.py`, `backend/app/api/health.py`
- Test: `backend/tests/test_registry.py`

**Interfaces:**
- Consumes: `Settings`; the three model wrappers from Tasks 9–11.
- Produces: `resolve_device(preference: str) -> str`; `ModelRegistry(settings, hand_factory=..., dinov2_factory=..., clip_factory=..., sam2_factory=...)` with `.gate()`, `.warm()`, `.detect_hand(image)`, `.embed(kind, image)`, `.segment(image, prompt)`, `.status() -> dict`; `AppState`, `get_state()`, `build_default_state()`.
- Health adds `models` and `device.resolved`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_registry.py
import numpy as np
import pytest

from app.config import Settings
from app.errors import ModelUnavailable
from app.ml.devices import resolve_device
from app.ml.registry import ModelRegistry


class FakeEmbedder:
    model_key = "dinov2-small"
    dim = 4

    def __init__(self, model_id=None, device="cpu"):
        self.device = device

    def embed(self, image):
        return np.ones(4, dtype=np.float32)


class FakeHand:
    def __init__(self, asset_path=None):
        self.asset_path = asset_path

    def detect(self, image):
        return None


def make_registry(tmp_path, **kwargs):
    settings = Settings(data_dir=tmp_path, device="cpu", warm_models=False)
    return ModelRegistry(
        settings,
        hand_factory=FakeHand,
        dinov2_factory=FakeEmbedder,
        clip_factory=FakeEmbedder,
        sam2_factory=lambda model_id, device: FakeEmbedder(model_id, device),
        **kwargs,
    )


def test_lazy_loading_and_status(tmp_path):
    registry = make_registry(tmp_path)
    assert registry.status()["models"]["dinov2"]["loaded"] is False
    vector = registry.embed("dinov2", np.zeros((8, 8, 3), dtype=np.uint8))
    assert vector.shape == (4,)
    assert registry.status()["models"]["dinov2"]["loaded"] is True
    assert registry.status()["device"]["resolved"] == "cpu"


def test_clip_is_unavailable_unless_enabled(tmp_path):
    registry = make_registry(tmp_path)
    with pytest.raises(ModelUnavailable):
        registry.embed("clip", np.zeros((8, 8, 3), dtype=np.uint8))


def test_mps_failure_falls_back_to_cpu(tmp_path):
    settings = Settings(data_dir=tmp_path, device="mps", warm_models=False)
    devices = []

    class FlakyEmbedder(FakeEmbedder):
        def __init__(self, model_id=None, device="cpu"):
            super().__init__(model_id, device)
            devices.append(device)
            self.calls = 0

        def embed(self, image):
            self.calls += 1
            if self.device == "mps" and self.calls == 1:
                raise RuntimeError("MPS kernel failed")
            return np.ones(4, dtype=np.float32)

    registry = ModelRegistry(
        settings,
        hand_factory=FakeHand,
        dinov2_factory=FlakyEmbedder,
        clip_factory=FakeEmbedder,
        sam2_factory=lambda model_id, device: FlakyEmbedder(model_id, device),
    )
    registry._device = "mps"  # deterministic test of the fallback branch
    vector = registry.embed("dinov2", np.zeros((8, 8, 3), dtype=np.uint8))
    assert vector.shape == (4,)
    assert devices == ["mps", "cpu"]
    assert registry.status()["models"]["dinov2"]["device"] == "cpu"
    assert "MPS kernel failed" in registry.status()["models"]["dinov2"]["error"]


def test_resolve_device_cpu_and_auto():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"mps", "cpu"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: app.ml`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/app/ml/__init__.py
```

```python
# backend/app/ml/devices.py
def resolve_device(preference: str) -> str:
    if preference == "cpu":
        return "cpu"
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
```

```python
# backend/app/ml/registry.py
from threading import RLock
from typing import Callable

from app.config import Settings
from app.errors import ModelUnavailable
from app.ml.devices import resolve_device
from app.pipeline.embedders import ClipEmbedder, DinoV2Embedder
from app.pipeline.hand import HandDetector
from app.pipeline.sam2 import Sam2Segmenter


class ModelRegistry:
    def __init__(
        self,
        settings: Settings,
        *,
        hand_factory: Callable = HandDetector,
        dinov2_factory: Callable = DinoV2Embedder,
        clip_factory: Callable = ClipEmbedder,
        sam2_factory: Callable = Sam2Segmenter,
    ):
        self.settings = settings
        self._gate = RLock()
        self._device = resolve_device(settings.device)
        self._force_cpu: set[str] = set()
        self._factories = {
            "hand": hand_factory,
            "dinov2": dinov2_factory,
            "clip": clip_factory,
            "sam2": sam2_factory,
        }
        self._hand = None
        self._embedders: dict[str, object] = {}
        self._sam2 = None
        self._errors: dict[str, str | None] = {
            "hand": None, "dinov2": None, "clip": None, "sam2": None
        }

    def gate(self) -> RLock:
        return self._gate

    def _device_for(self, name: str) -> str:
        return "cpu" if name in self._force_cpu else self._device

    def _ensure_hand(self) -> None:
        if self._hand is not None:
            return
        try:
            self._hand = self._factories["hand"](asset_path=self.settings.hand_asset)
            self._errors["hand"] = None
        except Exception as exc:
            self._errors["hand"] = str(exc)
            raise ModelUnavailable(str(exc)) from exc

    def _ensure_embedder(self, kind: str) -> object:
        if kind in self._embedders:
            return self._embedders[kind]
        if kind == "clip" and not self.settings.enable_clip:
            raise ModelUnavailable("CLIP is disabled (AUTO_SCANNER_ENABLE_CLIP=1 to enable)")
        key = "clip" if kind == "clip" else "dinov2"
        model_id = self.settings.clip_model_id if kind == "clip" else self.settings.dinov2_model_id
        try:
            instance = self._factories[key](model_id=model_id, device=self._device_for(key))
            self._embedders[kind] = instance
            self._errors[key] = None
            return instance
        except Exception as exc:
            self._errors[key] = str(exc)
            raise ModelUnavailable(str(exc)) from exc

    def _ensure_sam2(self):
        if self._sam2 is None:
            try:
                self._sam2 = self._factories["sam2"](
                    model_id=self.settings.sam2_model_id, device=self._device_for("sam2")
                )
                self._errors["sam2"] = None
            except Exception as exc:
                self._errors["sam2"] = str(exc)
                raise ModelUnavailable(str(exc)) from exc
        return self._sam2

    def warm(self) -> None:
        with self._gate:
            for name, ensure in (("hand", self._ensure_hand), ("dinov2", lambda: self._ensure_embedder("dinov2"))):
                try:
                    ensure()
                except ModelUnavailable:
                    pass

    def detect_hand(self, image):
        with self._gate:
            self._ensure_hand()
            return self._hand.detect(image)

    def embed(self, kind: str, image):
        with self._gate:
            return self._call_with_fallback(
                "clip" if kind == "clip" else "dinov2",
                lambda: self._ensure_embedder(kind).embed(image),
                reload=lambda: self._embedders.pop(kind, None),
            )

    def segment(self, image, prompt):
        with self._gate:
            return self._call_with_fallback(
                "sam2",
                lambda: self._ensure_sam2().segment(image, prompt),
                reload=lambda: setattr(self, "_sam2", None),
            )

    def _call_with_fallback(self, name: str, call, reload):
        try:
            return call()
        except Exception as exc:
            if self._device_for(name) != "mps":
                self._errors[name] = str(exc)
                raise
            message = str(exc)
            self._errors[name] = message
            self._force_cpu.add(name)
            reload()
            result = call()
            self._errors[name] = f"fell back to CPU: {message}"
            return result

    def status(self) -> dict:
        return {
            "device": {"preference": self.settings.device, "resolved": self._device},
            "models": {
                "hand": {"loaded": self._hand is not None, "device": "cpu", "error": self._errors["hand"]},
                "dinov2": {
                    "loaded": "dinov2" in self._embedders,
                    "device": self._device_for("dinov2"),
                    "error": self._errors["dinov2"],
                },
                "clip": {
                    "loaded": "clip" in self._embedders,
                    "device": self._device_for("clip"),
                    "error": self._errors["clip"],
                },
                "sam2": {
                    "loaded": self._sam2 is not None,
                    "device": self._device_for("sam2"),
                    "error": self._errors["sam2"],
                },
            },
        }
```

Modify `health.py`, `main.py`, and create `dependencies.py`:

```python
# backend/app/dependencies.py
from dataclasses import dataclass

from fastapi import Request

from app.config import Settings


@dataclass
class AppState:
    settings: Settings
    registry: object | None = None
    pipeline: object | None = None
    captures: object | None = None
    analysis: object | None = None
    evaluation: object | None = None


def get_state(request: Request) -> AppState:
    return request.app.state.services


def build_default_state(settings: Settings) -> AppState:
    from app.ml.registry import ModelRegistry

    return AppState(settings, registry=ModelRegistry(settings))
```

Tasks 13–16 extend `build_default_state` as each service is introduced:

```python
# after Task 13: pipeline = FramePipeline(registry, settings)
# after Task 14: captures = CaptureService(settings, pipeline)
# after Task 15: embeddings = EmbeddingService(settings, registry)
#                analysis = AnalysisService(settings, pipeline, embeddings)
# after Task 16: evaluation = EvaluationService(settings, embeddings)
```

```python
# backend/app/api/health.py
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict:
    settings = request.app.state.settings
    state = request.app.state.services
    body = {
        "status": "ok",
        "device": {"preference": settings.device},
        "assets": {"hand_landmarker": settings.hand_asset.exists()},
    }
    if state.registry is not None:
        registry_status = state.registry.status()
        body["device"].update(registry_status["device"])
        body["models"] = registry_status["models"]
    return body
```

```python
# backend/app/main.py (Task 12 state; later tasks add their routers)
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import bootstrap  # noqa: F401  (must run before torch)
from app.api.health import router as health_router
from app.api.items import router as items_router
from app.config import Settings, get_settings
from app.db import init_db
from app.dependencies import AppState, build_default_state


def create_app(
    settings: Settings | None = None, services: AppState | None = None
) -> FastAPI:
    settings = settings or get_settings()
    services = services or build_default_state(settings)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.media_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db(settings)
        if services.registry is not None and settings.warm_models:
            services.registry.warm()
        yield

    app = FastAPI(title="Supermarket Auto-Scanner", lifespan=lifespan)
    app.state.settings = settings
    app.state.services = services
    app.include_router(health_router, prefix="/api")
    app.include_router(items_router, prefix="/api")
    return app


app = create_app()
```

Later tasks extend this file: Task 14 adds the captures router and `/media` mount; Task 15 adds the analyze router; Task 16 adds the evaluations router and `recover_stale_runs(settings)` in lifespan; Task 17 adds the meta router.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_registry.py tests/test_health.py -v`
Expected: registry 4 passed + health 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/ml backend/app/dependencies.py backend/app/main.py backend/app/api/health.py backend/tests/test_registry.py
git commit -m "feat: lazy model registry with mps fallback and health status"
```

---

### Task 13: Variants, frame pipeline, annotation

**Files:**
- Create: `backend/app/pipeline/variants.py`, `backend/app/pipeline/frame_pipeline.py`, `backend/app/pipeline/annotate.py`
- Modify: `backend/app/dependencies.py`
- Test: `backend/tests/test_frame_pipeline.py`, `backend/tests/test_annotate.py`, `backend/tests/fakes.py`

**Interfaces:**
- Consumes: prompt geometry, mask selection, preprocess, registry, matcher types.
- Produces:
  - `Variant(id, label, crop, embedder)`, `VARIANTS`, `MODEL_KEYS`, `preprocess_variant(image, variant, localization, settings) -> np.ndarray | None`.
  - `PipelineOutput(localization, embeddings, variant_status, timings_ms, embed_ms)`.
  - `FramePipeline(registry, settings).localize(image) -> Localization` and `.analyze(image, variant_ids) -> PipelineOutput`.
  - `annotate_frame(image_rgb, localization, best, currency_symbol="$", label=None) -> bytes`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/fakes.py
import numpy as np

from app.pipeline.types import HandObservation, MaskCandidate, PromptSpec


class FakeHand:
    def __init__(self, asset_path=None):
        self.asset_path = asset_path
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        points = [(10.0, 10.0)] * 21
        points[0] = (10.0, 10.0)    # wrist
        points[4] = (50.0, 30.0)    # thumb tip
        points[5] = (12.0, 12.0)
        points[8] = (52.0, 30.0)    # index tip
        points[9] = (14.0, 14.0)
        points[12] = (55.0, 32.0)   # middle tip
        points[13] = (16.0, 16.0)
        points[17] = (18.0, 18.0)
        return HandObservation(
            landmarks_px=points,
            bbox=(10.0, 10.0, 55.0, 32.0),
            score=0.9,
        )


class NoHand:
    def __init__(self, asset_path=None):
        pass

    def detect(self, image):
        return None


class FakeSam2:
    def __init__(self, model_id=None, device="cpu", empty=False):
        self.empty = empty
        self.calls = 0

    def segment(self, image, prompt):
        self.calls += 1
        height, width = image.shape[:2]
        if self.empty:
            return [MaskCandidate(0, np.zeros((height, width), dtype=bool), 0.1)]
        mask = np.zeros((height, width), dtype=bool)
        mask[height // 4: 3 * height // 4, width // 4: 3 * width // 4] = True
        return [
            MaskCandidate(0, mask, 0.9),
            MaskCandidate(1, mask.copy(), 0.7),
            MaskCandidate(2, mask.copy(), 0.5),
        ]


class FakeEmbedder:
    dim = 8

    def __init__(self, model_id=None, device="cpu", constant=None):
        self.constant = constant

    def embed(self, image):
        if self.constant is not None:
            return np.asarray(self.constant, dtype=np.float32)
        value = float(np.mean(image)) / 255.0
        return np.full(self.dim, value, dtype=np.float32)


class RaisingEmbedder(FakeEmbedder):
    def embed(self, image):
        raise RuntimeError("embedding failed")


def make_settings(tmp_path):
    from app.config import Settings

    return Settings(data_dir=tmp_path, device="cpu", warm_models=False, enable_clip=True)
```

```python
# backend/tests/test_frame_pipeline.py
import numpy as np

from app.config import Settings
from app.pipeline.frame_pipeline import FramePipeline
from app.pipeline.types import PromptSpec
from tests.fakes import FakeEmbedder, FakeHand, FakeSam2, NoHand, make_settings


class Registry:
    def __init__(self, hand=None, segmenter=None, embedder=None):
        self.hand = hand or FakeHand()
        self.segmenter = segmenter or FakeSam2()
        self.embedder = embedder or FakeEmbedder()

    def gate(self):
        return __import__("threading").RLock()

    def detect_hand(self, image):
        return self.hand.detect(image)

    def segment(self, image, prompt):
        return self.segmenter.segment(image, prompt)

    def embed(self, kind, image):
        return self.embedder.embed(image)


def image():
    return np.full((100, 100, 3), 128, dtype=np.uint8)


def test_v2_runs_hand_sam2_and_dinov2(tmp_path):
    pipeline = FramePipeline(Registry(), make_settings(tmp_path))
    output = pipeline.analyze(image(), ["v0", "v1", "v2"])
    assert output.variant_status == {"v0": "ok", "v1": "ok", "v2": "ok"}
    assert output.localization.status == "ok"
    assert set(output.embeddings) == {"v0", "v1", "v2"}
    assert output.timings_ms["hand"] >= 0
    assert output.timings_ms["sam2"] >= 0


def test_no_hand_keeps_v0_and_reports_others(tmp_path):
    pipeline = FramePipeline(Registry(hand=NoHand()), make_settings(tmp_path))
    output = pipeline.analyze(image(), ["v0", "v1", "v2"])
    assert output.variant_status == {"v0": "ok", "v1": "no_hand", "v2": "no_hand"}
    assert output.localization.status == "no_hand"
    assert "v0" in output.embeddings and "v2" not in output.embeddings


def test_rejected_mask_falls_back_to_v1(tmp_path):
    pipeline = FramePipeline(
        Registry(segmenter=FakeSam2(empty=True)), make_settings(tmp_path)
    )
    output = pipeline.analyze(image(), ["v1", "v2"])
    assert output.localization.status == "mask_rejected"
    assert output.variant_status == {"v1": "ok", "v2": "mask_rejected"}


def test_v0_alone_never_runs_hand_or_sam2(tmp_path):
    hand, sam2 = FakeHand(), FakeSam2()
    pipeline = FramePipeline(Registry(hand=hand, segmenter=sam2), make_settings(tmp_path))
    pipeline.analyze(image(), ["v0"])
    assert hand.calls == 0 and sam2.calls == 0
```

```python
# backend/tests/test_annotate.py
from io import BytesIO

import numpy as np
from PIL import Image

from app.domain import TopItem
from app.pipeline.annotate import annotate_frame
from app.pipeline.frame_pipeline import FramePipeline
from tests.fakes import FakeHand, FakeSam2, make_settings


class Registry:
    def gate(self):
        import threading
        return threading.RLock()

    def detect_hand(self, image):
        return FakeHand().detect(image)

    def segment(self, image, prompt):
        return FakeSam2().segment(image, prompt)

    def embed(self, kind, image):
        return np.ones(8, dtype=np.float32)


def test_annotate_returns_decodable_jpeg(tmp_path):
    settings = make_settings(tmp_path)
    pipeline = FramePipeline(Registry(), settings)
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    output = pipeline.analyze(image, ["v2"])
    jpeg = annotate_frame(
        image,
        output.localization,
        TopItem(1, "Milk 1L", 199, 0.91),
        currency_symbol="$",
    )
    decoded = Image.open(BytesIO(jpeg))
    assert decoded.format == "JPEG"
    assert decoded.size == (100, 100)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && uv run pytest tests/test_frame_pipeline.py tests/test_annotate.py -v`
Expected: FAIL with `ModuleNotFoundError: app.pipeline.frame_pipeline`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/app/pipeline/variants.py
from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.config import Settings
from app.pipeline.preprocess import crop_box, crop_mask, letterbox
from app.pipeline.prompts import hand_crop_box
from app.pipeline.types import Localization

MODEL_KEYS = {"dinov2": "dinov2-small", "clip": "clip-vit-b32"}


@dataclass(frozen=True)
class Variant:
    id: str
    label: str
    crop: Literal["whole", "hand", "mask"]
    embedder: Literal["dinov2", "clip"]


VARIANTS: dict[str, Variant] = {
    "v0": Variant("v0", "Whole frame + DINOv2", "whole", "dinov2"),
    "v1": Variant("v1", "Hand crop + DINOv2", "hand", "dinov2"),
    "v2": Variant("v2", "SAM2 mask + DINOv2", "mask", "dinov2"),
    "v3": Variant("v3", "SAM2 mask + CLIP", "mask", "clip"),
}


def preprocess_variant(
    image_rgb: np.ndarray,
    variant: Variant,
    localization: Localization,
    settings: Settings,
) -> np.ndarray | None:
    height, width = image_rgb.shape[:2]
    if variant.crop == "whole":
        crop = image_rgb
    elif variant.crop == "hand":
        if localization.hand is None:
            return None
        box = hand_crop_box(
            localization.hand.landmarks_px, width, height, settings.prompt_box_expand
        )
        crop = crop_box(image_rgb, box, margin_ratio=0.0)
    else:
        if localization.mask is None:
            return None
        crop = crop_mask(image_rgb, localization.mask, margin_ratio=0.10)
    return letterbox(crop, settings.input_size)
```

```python
# backend/app/pipeline/frame_pipeline.py
import time
from dataclasses import dataclass, field

import numpy as np

from app.config import Settings
from app.errors import ModelUnavailable
from app.pipeline.mask_select import select_mask
from app.pipeline.prompts import build_prompt
from app.pipeline.types import Localization, MaskSelection
from app.pipeline.variants import VARIANTS, preprocess_variant


@dataclass(frozen=True)
class PipelineOutput:
    localization: Localization
    embeddings: dict[str, np.ndarray]
    variant_status: dict[str, str]
    timings_ms: dict[str, float]
    embed_ms: dict[str, float] = field(default_factory=dict)


class FramePipeline:
    def __init__(self, registry, settings: Settings):
        self.registry = registry
        self.settings = settings

    def localize(self, image_rgb: np.ndarray) -> Localization:
        with self.registry.gate():
            return self._localize(image_rgb)

    def _localize(self, image_rgb: np.ndarray) -> Localization:
        timings: dict[str, float] = {}
        start = time.perf_counter()
        hand = self.registry.detect_hand(image_rgb)
        timings["hand"] = (time.perf_counter() - start) * 1000

        if hand is None:
            return Localization(
                hand=None,
                prompt=None,
                selection=MaskSelection(None, [], [], []),
                mask=None,
                status="no_hand",
                timings_ms=timings,
            )

        height, width = image_rgb.shape[:2]
        prompt = build_prompt(
            hand.landmarks_px,
            width,
            height,
            mode=self.settings.prompt_mode,
            pinch_close_ratio=self.settings.pinch_close_ratio,
            box_expand=self.settings.prompt_box_expand,
        )
        start = time.perf_counter()
        candidates = self.registry.segment(image_rgb, prompt)
        timings["sam2"] = (time.perf_counter() - start) * 1000

        selection = select_mask(
            candidates,
            hand.landmarks_px,
            prompt.positive_points[0],
            prompt.negative_points,
            prompt.box,
            width * height,
            min_area_ratio=self.settings.mask_min_area_ratio,
            max_area_ratio=self.settings.mask_max_area_ratio,
            max_landmark_hits=self.settings.mask_max_landmark_hits,
        )
        if selection.selected is None:
            return Localization(
                hand=hand,
                prompt=prompt,
                selection=selection,
                mask=None,
                status="mask_rejected",
                timings_ms=timings,
            )
        return Localization(
            hand=hand,
            prompt=prompt,
            selection=selection,
            mask=selection.selected.mask,
            status="ok",
            timings_ms=timings,
        )

    def analyze(self, image_rgb: np.ndarray, variant_ids: list[str]) -> PipelineOutput:
        with self.registry.gate():
            return self._analyze(image_rgb, variant_ids)

    def _analyze(self, image_rgb: np.ndarray, variant_ids: list[str]) -> PipelineOutput:
        unknown = [v for v in variant_ids if v not in VARIANTS]
        if unknown:
            raise ValueError(f"Unknown variants: {unknown}")
        needs_localization = any(VARIANTS[v].crop in ("hand", "mask") for v in variant_ids)
        localization_error = None
        if needs_localization:
            try:
                localization = self._localize(image_rgb)
            except ModelUnavailable as exc:
                localization_error = str(exc)
                localization = Localization(
                    None, None, MaskSelection(None, [], [], []), None, "model_error"
                )
        else:
            localization = Localization(
                None, None, MaskSelection(None, [], [], []), None, "ok"
            )

        embeddings: dict[str, np.ndarray] = {}
        statuses: dict[str, str] = {}
        embed_ms: dict[str, float] = {}
        preprocess_ms = 0.0

        for variant_id in variant_ids:
            variant = VARIANTS[variant_id]
            if localization_error is not None and variant.crop in ("hand", "mask"):
                statuses[variant_id] = "model_error"
                continue
            if variant.embedder == "clip" and not self.settings.enable_clip:
                statuses[variant_id] = "model_error"
                continue
            if variant.crop == "hand" and localization.hand is None:
                statuses[variant_id] = "no_hand"
                continue
            if variant.crop == "mask" and localization.mask is None:
                statuses[variant_id] = "mask_rejected"
                continue
            preprocess_start = time.perf_counter()
            crop = preprocess_variant(image_rgb, variant, localization, self.settings)
            preprocess_ms += (time.perf_counter() - preprocess_start) * 1000
            if crop is None:
                statuses[variant_id] = "no_hand"
                continue
            embed_start = time.perf_counter()
            try:
                embeddings[variant_id] = self.registry.embed(variant.embedder, crop)
                statuses[variant_id] = "ok"
            except ModelUnavailable:
                statuses[variant_id] = "model_error"
            embed_ms[variant_id] = (time.perf_counter() - embed_start) * 1000

        timings = dict(localization.timings_ms)
        timings["preprocess"] = preprocess_ms
        timings["embed"] = sum(embed_ms.values())
        timings["total"] = sum(timings.values())
        return PipelineOutput(
            localization=localization,
            embeddings=embeddings,
            variant_status=statuses,
            timings_ms=timings,
            embed_ms=embed_ms,
        )
```

```python
# backend/app/pipeline/annotate.py
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

from app.domain import TopItem
from app.pipeline.types import Localization


def annotate_frame(
    image_rgb: np.ndarray,
    localization: Localization,
    best: TopItem | None,
    currency_symbol: str = "$",
    label: str | None = None,
) -> bytes:
    image = Image.fromarray(image_rgb.copy()).convert("RGB")
    if localization.mask is not None:
        array = np.asarray(image).astype(np.float32)
        array[localization.mask] = array[localization.mask] * 0.55 + np.array([0, 120, 255]) * 0.45
        image = Image.fromarray(array.clip(0, 255).astype(np.uint8))

    draw = ImageDraw.Draw(image)
    if localization.hand is not None:
        draw.rectangle(localization.hand.bbox, outline=(0, 220, 0), width=3)
    if localization.prompt is not None:
        for x, y in localization.prompt.positive_points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 0, 0))
        for x, y in localization.prompt.negative_points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 200, 0))

    text = label
    if text is None and best is not None:
        text = f"{best.name}  {currency_symbol}{best.price_cents / 100:.2f}  ({best.score:.2f})"
    if text:
        draw.rectangle((4, 4, 4 + 7 * len(text), 24), fill=(0, 0, 0))
        draw.text((8, 8), text, fill=(255, 255, 255))

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    return buffer.getvalue()
```

Extend `build_default_state` in `dependencies.py`:

```python
from app.pipeline.frame_pipeline import FramePipeline

def build_default_state(settings: Settings) -> AppState:
    from app.ml.registry import ModelRegistry

    registry = ModelRegistry(settings)
    pipeline = FramePipeline(registry, settings)
    return AppState(settings, registry=registry, pipeline=pipeline)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_frame_pipeline.py tests/test_annotate.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/variants.py backend/app/pipeline/frame_pipeline.py backend/app/pipeline/annotate.py backend/tests/fakes.py backend/tests/test_frame_pipeline.py backend/tests/test_annotate.py
git commit -m "feat: frame pipeline, variants, and annotation"
```

---

### Task 14: Capture upload service and captures API

**Files:**
- Modify: `backend/app/services/captures.py`, `backend/app/schemas.py`, `backend/app/main.py`, `backend/app/dependencies.py`
- Create: `backend/app/api/captures.py`
- Test: `backend/tests/test_captures_api.py`

**Interfaces:**
- Consumes: `FramePipeline.localize`, storage, capture store, `AppState`.
- Produces: `CaptureService(settings, pipeline)` with `.add(item_id, role, session_id, data) -> CaptureUploadResult`, `.list(item_id, role=None)`, `.delete(item_id, capture_id)`; `localization_to_dict(localization) -> dict`; `load_capture_localization(settings, capture) -> Localization`; `POST/GET/DELETE /api/items/{item_id}/captures`; `/media` static mount.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_captures_api.py
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.dependencies import AppState, build_default_state
from app.main import create_app
from app.services.captures import CaptureService
from tests.fakes import FakeHand, FakeSam2, NoHand, make_settings
from app.pipeline.frame_pipeline import FramePipeline


class Registry:
    def __init__(self, hand=None):
        self.hand = hand or FakeHand()
        self.sam2 = FakeSam2()

    def gate(self):
        import threading
        return threading.RLock()

    def detect_hand(self, image):
        return self.hand.detect(image)

    def segment(self, image, prompt):
        return self.sam2.segment(image, prompt)

    def embed(self, kind, image):
        import numpy as np
        return np.ones(8, dtype=np.float32)


def jpeg_bytes(color=(200, 40, 40)):
    buffer = BytesIO()
    Image.new("RGB", (64, 64), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def make_app(tmp_path, hand=None):
    settings = make_settings(tmp_path)
    pipeline = FramePipeline(Registry(hand), settings)
    services = AppState(settings, pipeline=pipeline, captures=CaptureService(settings, pipeline))
    return create_app(settings, services), settings


def test_upload_accepts_valid_capture_and_serves_media(tmp_path):
    app, _ = make_app(tmp_path)
    with TestClient(app) as client:
        item = client.post("/api/items", json={"name": "Milk", "price_cents": 199}).json()
        response = client.post(
            f"/api/items/{item['id']}/captures?role=gallery&session_id=G1",
            files={"files": ("frame.jpg", jpeg_bytes(), "image/jpeg")},
        )
        assert response.status_code == 201
        body = response.json()
        assert len(body["accepted"]) == 1
        capture = body["accepted"][0]
        assert capture["mask_status"] == "ok"
        assert client.get(f"/media/{capture['thumb_rel_path']}").status_code == 200
        listed = client.get(f"/api/items/{item['id']}/captures?role=gallery").json()
        assert len(listed) == 1


def test_upload_rejects_no_hand(tmp_path):
    app, _ = make_app(tmp_path, hand=NoHand())
    with TestClient(app) as client:
        item = client.post("/api/items", json={"name": "Milk", "price_cents": 199}).json()
        response = client.post(
            f"/api/items/{item['id']}/captures?role=gallery&session_id=G1",
            files={"files": ("frame.jpg", jpeg_bytes(), "image/jpeg")},
        )
        assert response.status_code == 201
        assert response.json()["accepted"] == []
        assert response.json()["rejected"][0]["reason"] == "no_hand"
        assert client.get(f"/api/items/{item['id']}/captures").json() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_captures_api.py -v`
Expected: FAIL because `CaptureService` and routes are missing.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/app/schemas.py (append)
class CaptureOut(BaseModel):
    id: int
    item_id: int
    role: str
    session_id: str
    rel_path: str
    thumb_rel_path: str
    mask_rel_path: str
    width: int
    height: int
    hand_found: bool
    mask_found: bool
    mask_status: str


class CaptureAccepted(BaseModel):
    capture: CaptureOut
    thumb_rel_path: str
    mask_rel_path: str
    mask_status: str


class CaptureRejected(BaseModel):
    reason: str


class CaptureUploadResponse(BaseModel):
    accepted: list[CaptureAccepted]
    rejected: list[CaptureRejected]
```

```python
# backend/app/services/captures.py (append)
from dataclasses import dataclass

from app.errors import DuplicateCapture, ImageTooLarge, InvalidImage, ModelUnavailable
from app.pipeline.types import HandObservation, Localization, MaskSelection, PromptSpec
from app.services.storage import (
    decode_image, load_mask, save_capture_bundle, sha256_hex,
)


@dataclass(frozen=True)
class CaptureUploadResult:
    accepted: bool
    capture: Capture | None = None
    reason: str | None = None


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


def load_capture_localization(settings, capture: Capture) -> Localization:
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


def capture_duplicate_exists(settings, item_id: int, role: str, sha256: str) -> bool:
    with db(settings) as connection:
        row = connection.execute(
            "SELECT 1 FROM captures WHERE item_id = ? AND role = ? AND sha256 = ?",
            (item_id, role, sha256),
        ).fetchone()
    return row is not None


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
        return CaptureUploadResult(True, capture=capture)

    def list(self, item_id: int, role: str | None = None):
        return list_captures(self.settings, item_id, role)  # type: ignore[arg-type]

    def delete(self, item_id: int, capture_id: int) -> bool:
        return delete_capture(self.settings, item_id, capture_id)
```

```python
# backend/app/api/captures.py
from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile, status

from app.dependencies import get_state
from app.schemas import CaptureOut, CaptureRejected, CaptureUploadResponse, CaptureAccepted
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
    request: Request,
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
        result = state.captures.add(item_id, role, session_id, data)
        if result.accepted and result.capture is not None:
            accepted.append(
                CaptureAccepted(
                    capture=_capture_out(result.capture),
                    thumb_rel_path=result.capture.thumb_rel_path,
                    mask_rel_path=result.capture.mask_rel_path,
                    mask_status=result.capture.mask_status,
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
```

Mount media and include the router in `main.py`:

```python
from fastapi.staticfiles import StaticFiles

from app.api import captures as captures_api

app.mount("/media", StaticFiles(directory=settings.media_dir), name="media")
app.include_router(captures_api.router, prefix="/api")
```

Extend `build_default_state` in `dependencies.py`:

```python
from app.services.captures import CaptureService

def build_default_state(settings: Settings) -> AppState:
    from app.ml.registry import ModelRegistry
    from app.pipeline.frame_pipeline import FramePipeline

    registry = ModelRegistry(settings)
    pipeline = FramePipeline(registry, settings)
    return AppState(
        settings, registry=registry, pipeline=pipeline,
        captures=CaptureService(settings, pipeline),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_captures_api.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/captures.py backend/app/schemas.py backend/app/api/captures.py backend/app/main.py backend/tests/test_captures_api.py
git commit -m "feat: webcam capture upload and media API"
```

---

### Task 15: Lazy embedding cache, analysis service, analyze API

**Files:**
- Create: `backend/app/services/embeddings.py`, `backend/app/services/analysis.py`, `backend/app/api/analyze.py`
- Modify: `backend/app/schemas.py`, `backend/app/main.py`, `backend/app/dependencies.py`
- Test: `backend/tests/test_analyze_api.py`

**Interfaces:**
- Consumes: `FramePipeline`, capture store, vector matcher, annotate.
- Produces: `EmbeddingService(settings, registry)` with `.get_capture_vector(capture, variant_id) -> np.ndarray | None` and `.gallery_entries(variant_id) -> list[GalleryEntry]`; `AnalysisService(settings, pipeline, embeddings)` with `.analyze(image_bytes, variant_ids) -> AnalyzeResult`; `POST /api/analyze` returns `{frame_id, annotated_jpeg_b64, hand_found, mask_found, mask_status, variants:[{variant,status,top_items,timings_ms,error}]}`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_analyze_api.py
from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.dependencies import AppState
from app.matching.matcher import GalleryEntry
from app.services.analysis import AnalysisService, AnalyzeResult, VariantAnalysis
from app.domain import TopItem
from app.main import create_app
from tests.fakes import make_settings


class FakePipeline:
    def analyze(self, image, variant_ids):
        from app.pipeline.frame_pipeline import PipelineOutput
        from app.pipeline.types import Localization, MaskSelection
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[10:40, 10:40] = True
        localization = Localization(None, None, MaskSelection(None, [], [], []), mask, "ok", {})
        return PipelineOutput(
            localization=localization,
            embeddings={v: np.array([1, 0, 0, 0], dtype=np.float32) for v in variant_ids},
            variant_status={v: "ok" for v in variant_ids},
            timings_ms={"hand": 1.0, "sam2": 2.0, "preprocess": 3.0, "embed": 4.0, "total": 10.0},
            embed_ms={v: 4.0 for v in variant_ids},
        )


class FakeEmbeddings:
    def gallery_entries(self, variant_id):
        return [
            GalleryEntry(1, "Milk 1L", 199, np.array([[1, 0, 0, 0]], dtype=np.float32)),
            GalleryEntry(2, "Bread", 250, np.array([[0, 1, 0, 0]], dtype=np.float32)),
        ]


def jpeg_bytes():
    buffer = BytesIO()
    Image.new("RGB", (64, 64), (100, 100, 100)).save(buffer, format="JPEG")
    return buffer.getvalue()


def make_app(tmp_path):
    settings = make_settings(tmp_path)
    analysis = AnalysisService(settings, FakePipeline(), FakeEmbeddings())
    services = AppState(settings, analysis=analysis)
    return create_app(settings, services)


def test_analyze_returns_top3_and_annotated_image(tmp_path):
    with TestClient(make_app(tmp_path)) as client:
        response = client.post(
            "/api/analyze",
            files={"image": ("frame.jpg", jpeg_bytes(), "image/jpeg")},
            data={"variants": "v0,v2"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["hand_found"] is False
    assert len(body["variants"]) == 2
    assert body["variants"][0]["top_items"][0]["name"] == "Milk 1L"
    assert body["variants"][0]["top_items"][0]["price_cents"] == 199
    assert body["annotated_jpeg_b64"]


def test_analyze_with_empty_gallery_reports_no_gallery(tmp_path):
    settings = make_settings(tmp_path)
    analysis = AnalysisService(settings, FakePipeline(), FakeEmbeddingsNoGallery())
    with TestClient(create_app(settings, AppState(settings, analysis=analysis))) as client:
        body = client.post(
            "/api/analyze",
            files={"image": ("frame.jpg", jpeg_bytes(), "image/jpeg")},
            data={"variants": "v2"},
        ).json()
    assert body["variants"][0]["status"] == "no_gallery"


class FakeEmbeddingsNoGallery(FakeEmbeddings):
    def gallery_entries(self, variant_id):
        return []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_analyze_api.py -v`
Expected: FAIL with `ModuleNotFoundError: app.services.analysis`.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/app/services/embeddings.py
import numpy as np

from app.matching.matcher import GalleryEntry
from app.pipeline.preprocess import PREPROCESS_VERSION
from app.pipeline.variants import MODEL_KEYS, VARIANTS, preprocess_variant
from app.services.captures import (
    get_embedding,
    list_gallery_captures,
    load_capture_localization,
    upsert_embedding,
)
from app.services.storage import load_image_rgb


class EmbeddingService:
    def __init__(self, settings, registry):
        self.settings = settings
        self.registry = registry

    def get_capture_vector(self, capture, variant_id: str) -> np.ndarray | None:
        variant = VARIANTS[variant_id]
        model_key = MODEL_KEYS[variant.embedder]
        cached = get_embedding(
            self.settings, capture.id, variant_id, model_key, PREPROCESS_VERSION
        )
        if cached is not None:
            return cached

        image = load_image_rgb(self.settings, capture.rel_path)
        localization = load_capture_localization(self.settings, capture)
        crop = preprocess_variant(image, variant, localization, self.settings)
        if crop is None:
            return None
        vector = self.registry.embed(variant.embedder, crop)
        upsert_embedding(
            self.settings, capture.id, variant_id, model_key, PREPROCESS_VERSION, vector
        )
        return vector

    def gallery_entries(self, variant_id: str) -> list[GalleryEntry]:
        grouped: dict[int, dict] = {}
        for capture, name, price_cents in list_gallery_captures(self.settings):
            vector = self.get_capture_vector(capture, variant_id)
            if vector is None:
                continue
            bucket = grouped.setdefault(
                capture.item_id, {"name": name, "price": price_cents, "vectors": []}
            )
            bucket["vectors"].append(vector)
        return [
            GalleryEntry(item_id, data["name"], data["price"], np.vstack(data["vectors"]))
            for item_id, data in grouped.items()
        ]


# add to services/captures.py
def list_gallery_captures(settings):
    with db(settings) as connection:
        rows = connection.execute(
            """
            SELECT c.*, i.name, i.price_cents FROM captures c
            JOIN items i ON i.id = c.item_id
            WHERE c.role = 'gallery' ORDER BY c.item_id, c.id
            """
        ).fetchall()
    return [(_row_to_capture(row), row["name"], row["price_cents"]) for row in rows]
```

```python
# backend/app/services/analysis.py
import time
import uuid
from dataclasses import dataclass, field

from app.domain import TopItem
from app.errors import ImageTooLarge, InvalidImage, ModelUnavailable
from app.matching.matcher import item_topk
from app.pipeline.annotate import annotate_frame
from app.pipeline.variants import VARIANTS
from app.services.storage import decode_image


@dataclass(frozen=True)
class VariantAnalysis:
    variant: str
    status: str
    top_items: list[TopItem] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class AnalyzeResult:
    frame_id: str
    annotated_jpeg: bytes
    hand_found: bool
    mask_found: bool
    mask_status: str
    variants: list[VariantAnalysis]


class AnalysisService:
    def __init__(self, settings, pipeline, embeddings):
        self.settings = settings
        self.pipeline = pipeline
        self.embeddings = embeddings

    def analyze(self, image_bytes: bytes, variant_ids: list[str]) -> AnalyzeResult:
        image, _, _ = decode_image(image_bytes, self.settings.max_upload_bytes)
        output = self.pipeline.analyze(image, variant_ids)
        variants: list[VariantAnalysis] = []

        for variant_id in variant_ids:
            status = output.variant_status.get(variant_id, "model_error")
            top_items: list[TopItem] = []
            match_ms = 0.0
            error = None
            if status == "ok":
                match_start = time.perf_counter()
                entries = self.embeddings.gallery_entries(variant_id)
                if not entries:
                    status = "no_gallery"
                else:
                    top_items = item_topk(output.embeddings[variant_id], entries, k=3)
                match_ms = (time.perf_counter() - match_start) * 1000
            timings = {
                **{key: value for key, value in output.timings_ms.items() if key != "embed"},
                "embed": output.embed_ms.get(variant_id, 0.0),
                "match": match_ms,
            }
            timings["total"] = sum(timings.values())
            variants.append(VariantAnalysis(variant_id, status, top_items, timings, error))

        best = max(
            (v.top_items[0] for v in variants if v.top_items),
            key=lambda item: item.score,
            default=None,
        )
        annotated = annotate_frame(
            image, output.localization, best, self.settings.currency_symbol
        )
        return AnalyzeResult(
            frame_id=uuid.uuid4().hex,
            annotated_jpeg=annotated,
            hand_found=output.localization.hand is not None,
            mask_found=output.localization.mask is not None,
            mask_status=output.localization.status,
            variants=variants,
        )
```

```python
# backend/app/schemas.py (append)
class TopItemOut(BaseModel):
    item_id: int
    name: str
    price_cents: int
    score: float


class VariantOut(BaseModel):
    variant: str
    status: str
    top_items: list[TopItemOut]
    timings_ms: dict[str, float]
    error: str | None = None


class AnalyzeOut(BaseModel):
    frame_id: str
    annotated_jpeg_b64: str
    hand_found: bool
    mask_found: bool
    mask_status: str
    variants: list[VariantOut]
```

```python
# backend/app/api/analyze.py
import base64

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.dependencies import get_state
from app.errors import ImageTooLarge, InvalidImage, ModelUnavailable
from app.schemas import AnalyzeOut, TopItemOut, VariantOut

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeOut)
async def analyze(
    image: UploadFile = File(...),
    variants: str = Form("v0,v1,v2"),
    state=Depends(get_state),
):
    if state.analysis is None:
        raise HTTPException(status_code=503, detail="Analysis service unavailable")
    variant_ids = [v.strip() for v in variants.split(",") if v.strip()]
    if not variant_ids:
        raise HTTPException(status_code=422, detail="At least one variant is required")
    data = await image.read()
    try:
        result = state.analysis.analyze(data, variant_ids)
    except ImageTooLarge as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidImage as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if result.variants and all(v.status == "model_error" for v in result.variants):
        raise HTTPException(status_code=503, detail="All requested models are unavailable")

    return AnalyzeOut(
        frame_id=result.frame_id,
        annotated_jpeg_b64=base64.b64encode(result.annotated_jpeg).decode(),
        hand_found=result.hand_found,
        mask_found=result.mask_found,
        mask_status=result.mask_status,
        variants=[
            VariantOut(
                variant=v.variant,
                status=v.status,
                top_items=[TopItemOut(**item.__dict__) for item in v.top_items],
                timings_ms=v.timings_ms,
                error=v.error,
            )
            for v in result.variants
        ],
    )
```

Wire the router and default analysis service in `main.py`/`dependencies.py`:

```python
# dependencies.py build_default_state additions
from app.services.analysis import AnalysisService
from app.services.embeddings import EmbeddingService

embeddings = EmbeddingService(settings, registry)
return AppState(
    settings, registry, pipeline, CaptureService(settings, pipeline),
    AnalysisService(settings, pipeline, embeddings),
)
```

```python
# main.py
from app.api import analyze as analyze_api

# in create_app:
app.include_router(analyze_api.router, prefix="/api")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_analyze_api.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/embeddings.py backend/app/services/analysis.py backend/app/api/analyze.py backend/app/schemas.py backend/app/main.py backend/app/dependencies.py backend/tests/test_analyze_api.py
git commit -m "feat: analyze API with cached gallery embeddings"
```

---

### Task 16: Evaluation runner, artifacts, and evaluations API

**Files:**
- Create: `backend/app/evaluation/artifacts.py`, `backend/app/evaluation/runner.py`, `backend/app/services/evaluation.py`, `backend/app/api/evaluations.py`
- Modify: `backend/app/db.py`, `backend/app/dependencies.py`, `backend/app/main.py`, `backend/app/schemas.py`
- Test: `backend/tests/test_evaluation_runner.py`, `backend/tests/test_evaluations_api.py`

**Interfaces:**
- Consumes: `EmbeddingService`, capture store, metrics, matcher, artifacts.
- Produces: `recover_stale_runs(settings)`; `EvaluationRunner(settings, embeddings).run(run_id, variant_ids, item_ids, progress) -> dict`; `EvaluationService(settings, embeddings)` with `.start(variants, item_ids=None, background=True) -> str`, `.get(run_id) -> dict`, `.list()`; `POST /api/evaluations` (202), `GET /api/evaluations`, `GET /api/evaluations/{run_id}`, `GET /api/evaluations/{run_id}/failures`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_evaluation_runner.py
import numpy as np

from app.config import Settings
from app.db import init_db
from app.evaluation.metrics import SampleResult
from app.matching.matcher import GalleryEntry
from app.services.captures import add_capture
from app.services.items import create_item
from app.services.storage import save_capture_bundle


class FakeEmbeddings:
    def __init__(self):
        self.vectors = {}  # capture_id -> vector

    def gallery_entries(self, variant_id):
        rows = self.rows
        grouped = {}
        for row in rows:
            if row["role"] != "gallery":
                continue
            grouped.setdefault(row["item_id"], []).append(row["vector"])
        names = self.names
        return [
            GalleryEntry(item_id, names[item_id][0], names[item_id][1], np.vstack(vectors))
            for item_id, vectors in grouped.items()
        ]

    def get_capture_vector(self, capture, variant_id):
        return self.vectors[capture.id]


def seed(tmp_path):
    settings = Settings(data_dir=tmp_path, warm_models=False)
    init_db(settings)
    fake = FakeEmbeddings()
    fake.rows, fake.names = [], {}
    for idx, (name, vector) in enumerate(
        [("Milk", [1, 0]), ("Bread", [0, 1])], start=1
    ):
        item = create_item(settings, name, idx * 100)
        fake.names[item.id] = (name, idx * 100)
        image = np.full((16, 16, 3), 100 + idx, dtype=np.uint8)
        mask = np.ones((16, 16), dtype=bool)
        for role in ("gallery", "test"):
            artifacts = save_capture_bundle(settings, item.id, image, mask)
            capture = add_capture(
                settings, item_id=item.id, role=role, session_id="s",
                artifacts=artifacts, sha256=f"{item.id}-{role}".ljust(64, "0"),
                width=16, height=16, hand_found=True, mask_found=True,
                mask_status="ok", localization={"status": "ok", "timings_ms": {}},
            )
            fake.vectors[capture.id] = np.array(vector, dtype=np.float32)
            fake.rows.append({"item_id": item.id, "role": role, "vector": fake.vectors[capture.id]})
    return settings, fake


def test_runner_computes_perfect_accuracy(tmp_path):
    from app.evaluation.runner import EvaluationRunner

    settings, fake = seed(tmp_path)
    run_id = "run-1"
    from app.services.evaluation import create_run
    create_run(settings, run_id, {"variants": ["v2"]})
    metrics = EvaluationRunner(settings, fake).run(run_id, ["v2"], None, lambda done, total: None)
    assert metrics["v2"]["n"] == 2
    assert metrics["v2"]["top1"] == 1.0
```

```python
# backend/tests/test_evaluations_api.py
from fastapi.testclient import TestClient

from app.dependencies import AppState
from app.main import create_app
from tests.fakes import make_settings


class FakeEvaluationService:
    def start(self, variants, item_ids=None):
        return "run-1"

    def list(self):
        return [{"id": "run-1", "status": "completed"}]

    def get(self, run_id):
        if run_id != "run-1":
            return None
        return {
            "id": "run-1", "status": "completed", "metrics": {},
            "failures": [{"variant": "v2", "true_item_name": "Milk"}],
        }


def test_start_and_get_evaluation(tmp_path):
    settings = make_settings(tmp_path)
    services = AppState(settings, evaluation=FakeEvaluationService())
    with TestClient(create_app(settings, services)) as client:
        started = client.post("/api/evaluations", json={"variants": ["v2"]})
        assert started.status_code == 202
        assert started.json()["run_id"] == "run-1"
        assert client.get("/api/evaluations/run-1").status_code == 200
        assert client.get("/api/evaluations/missing").status_code == 404
        failures = client.get("/api/evaluations/run-1/failures?variant=v2").json()
        assert failures[0]["variant"] == "v2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_evaluation_runner.py tests/test_evaluations_api.py -v`
Expected: FAIL with `ModuleNotFoundError: app.evaluation.runner` / missing routes.

- [ ] **Step 3: Write minimal implementation**

```python
# backend/app/evaluation/artifacts.py
from pathlib import Path

from app.domain import TopItem
from app.pipeline.annotate import annotate_frame
from app.services.storage import resolve_media_path


def save_failure_artifact(
    settings, run_id: str, key: str, image_rgb, localization,
    true_name: str, predicted: TopItem | None,
) -> str:
    label = f"true: {true_name} | pred: {predicted.name if predicted else 'none'}"
    jpeg = annotate_frame(
        image_rgb, localization, None, settings.currency_symbol, label=label
    )
    rel_path = f"runs/{run_id}/{key}.jpg"
    path: Path = resolve_media_path(settings, rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(jpeg)
    return rel_path
```

```python
# backend/app/evaluation/runner.py
import json
import time
from datetime import datetime, timezone

import numpy as np

from app.evaluation.artifacts import save_failure_artifact
from app.evaluation.metrics import SampleResult, compute_metrics
from app.matching.matcher import item_topk
from app.matching.vector_index import normalize
from app.services.captures import list_test_captures, load_capture_localization
from app.services.storage import load_image_rgb
from app.db import db


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class EvaluationRunner:
    def __init__(self, settings, embeddings):
        self.settings = settings
        self.embeddings = embeddings

    def run(self, run_id, variant_ids, item_ids, progress) -> dict:
        captures = list_test_captures(self.settings, item_ids)
        gallery = {variant: self.embeddings.gallery_entries(variant) for variant in variant_ids}
        samples: dict[str, list[SampleResult]] = {variant: [] for variant in variant_ids}
        near_duplicates = {variant: 0 for variant in variant_ids}
        total = len(captures) * len(variant_ids)
        done = 0

        for capture in captures:
            image = load_image_rgb(self.settings, capture.rel_path)
            localization = load_capture_localization(self.settings, capture)
            localization_ms = capture.localization.get("timings_ms", {})
            for variant in variant_ids:
                embed_ms = 0.0
                match_ms = 0.0
                status = "ok"
                try:
                    embed_start = time.perf_counter()
                    vector = self.embeddings.get_capture_vector(capture, variant)
                    embed_ms = (time.perf_counter() - embed_start) * 1000
                    if vector is None:
                        status = "mask_rejected"
                        top = []
                    else:
                        entries = gallery[variant]
                        if not entries:
                            status = "no_gallery"
                            top = []
                        else:
                            if any(
                                entry.vectors.size
                                and float(np.max(entry.vectors @ normalize(vector))) > 0.995
                                for entry in entries
                            ):
                                near_duplicates[variant] += 1
                            match_start = time.perf_counter()
                            top = item_topk(vector, entries, k=5)
                            match_ms = (time.perf_counter() - match_start) * 1000
                except Exception:
                    status = "model_error"
                    top = []
                timings = {
                    **localization_ms, "embed": embed_ms, "match": match_ms,
                    "total": embed_ms + match_ms + sum(localization_ms.values()),
                }
                sample = SampleResult(
                    true_item_id=capture.item_id,
                    true_item_name=capture.item_name,
                    true_price_cents=capture.item_price_cents,
                    top=top,
                    status=status,
                    timings_ms=timings,
                )
                samples[variant].append(sample)
                correct1 = bool(top and top[0].item_id == capture.item_id)
                correct5 = any(item.item_id == capture.item_id for item in top[:5])
                artifact = None
                if status != "ok" or not correct1:
                    artifact = save_failure_artifact(
                        self.settings, run_id, f"{capture.id}-{variant}",
                        image, localization, capture.item_name,
                        top[0] if top else None,
                    )
                with db(self.settings) as connection:
                    connection.execute(
                        """
                        INSERT INTO eval_samples
                          (run_id, capture_id, variant, true_item_id, true_item_name,
                           true_price_cents, pred_item_id, pred_item_name, topk_json,
                           correct_top1, correct_top5, sample_status, timings_json,
                           rendered_rel_path, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id, capture.id, variant, capture.item_id, capture.item_name,
                            capture.item_price_cents,
                            top[0].item_id if top else None,
                            top[0].name if top else None,
                            json.dumps([item.__dict__ for item in top]),
                            int(correct1), int(correct5), status, json.dumps(timings),
                            artifact, _now(),
                        ),
                    )
                done += 1
                progress(done, total)

        metrics = {
            variant: compute_metrics(samples[variant], n_items=len({c.item_id for c in captures}))
            for variant in variant_ids
        }
        for variant in variant_ids:
            metrics[variant]["near_duplicate_count"] = near_duplicates[variant]
        return metrics
```

Add `list_test_captures` to `services/captures.py` and extend the returned captures with item name/price. Since `Capture` dataclass doesn't have them, define a small `EvalCapture` dataclass in `services/captures.py`:

```python
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


def list_test_captures(settings, item_ids=None) -> list[EvalCapture]:
    query = """
      SELECT c.*, i.name AS item_name, i.price_cents AS item_price_cents
      FROM captures c JOIN items i ON i.id = c.item_id
      WHERE c.role = 'test'
    """
    params = []
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
```

`load_capture_localization` expects a `Capture` with `.localization`, `.mask_rel_path`, `.mask_found`; `EvalCapture` satisfies those duck-typed attributes.

```python
# backend/app/services/evaluation.py
import json
import threading
import uuid
from datetime import datetime, timezone

from app.db import db
from app.errors import ModelUnavailable
from app.evaluation.runner import EvaluationRunner
from app.services.captures import list_test_captures
from app.services.items import list_items


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_run(settings, run_id, request, progress_total=0):
    with db(settings) as connection:
        connection.execute(
            """
            INSERT INTO eval_runs
              (id, created_at, status, request_json, progress_done, progress_total)
            VALUES (?, ?, 'queued', ?, 0, ?)
            """,
            (run_id, _now(), json.dumps(request), progress_total),
        )


def recover_stale_runs(settings):
    with db(settings) as connection:
        connection.execute(
            """
            UPDATE eval_runs SET status = 'failed', error = 'interrupted by restart'
            WHERE status IN ('queued', 'running')
            """
        )


class EvaluationService:
    def __init__(self, settings, embeddings):
        self.settings = settings
        self.embeddings = embeddings
        self._thread = None

    def _validate(self, variant_ids, item_ids):
        from app.pipeline.variants import VARIANTS

        if not variant_ids or any(variant not in VARIANTS for variant in variant_ids):
            raise ValueError("Select at least one valid variant")
        if "v3" in variant_ids and not self.settings.enable_clip:
            raise ValueError("V3 requires CLIP (AUTO_SCANNER_ENABLE_CLIP=1)")
        items = list_items(self.settings)
        if item_ids:
            items = [item for item in items if item.id in set(item_ids)]
        usable = [
            item for item in items
            if item.gallery_count >= 1 and item.test_count >= 1
        ]
        if len(usable) < 2:
            raise ValueError("Need at least 2 items with both gallery and test captures")
        if len(list_test_captures(self.settings, item_ids)) == 0:
            raise ValueError("No test captures found")
        return usable

    def start(self, variant_ids, item_ids=None, background=True) -> str:
        self._validate(variant_ids, item_ids)
        if self._thread and self._thread.is_alive():
            raise RuntimeError("Evaluation already running")
        run_id = uuid.uuid4().hex
        request = {"variants": variant_ids, "item_ids": item_ids}
        create_run(self.settings, run_id, request)
        if background:
            self._thread = threading.Thread(
                target=self._run, args=(run_id, variant_ids, item_ids), daemon=True
            )
            self._thread.start()
        else:
            self._run(run_id, variant_ids, item_ids)
        return run_id

    def _run(self, run_id, variant_ids, item_ids):
        with db(self.settings) as connection:
            connection.execute(
                "UPDATE eval_runs SET status='running', started_at=? WHERE id=?",
                (_now(), run_id),
            )

        def progress(done, total):
            with db(self.settings) as connection:
                connection.execute(
                    "UPDATE eval_runs SET progress_done=?, progress_total=? WHERE id=?",
                    (done, total, run_id),
                )

        try:
            metrics = EvaluationRunner(self.settings, self.embeddings).run(
                run_id, variant_ids, item_ids, progress
            )
            with db(self.settings) as connection:
                connection.execute(
                    """
                    UPDATE eval_runs SET status='completed', finished_at=?,
                      metrics_json=?, progress_done=progress_total WHERE id=?
                    """,
                    (_now(), json.dumps(metrics), run_id),
                )
        except Exception as exc:
            with db(self.settings) as connection:
                connection.execute(
                    "UPDATE eval_runs SET status='failed', error=?, finished_at=? WHERE id=?",
                    (str(exc), _now(), run_id),
                )

    def get(self, run_id):
        with db(self.settings) as connection:
            row = connection.execute("SELECT * FROM eval_runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                return None
            failures = connection.execute(
                """
                SELECT * FROM eval_samples
                WHERE run_id=? AND (correct_top1=0 OR sample_status!='ok')
                ORDER BY id LIMIT 50
                """,
                (run_id,),
            ).fetchall()
        result = dict(row)
        result["metrics"] = json.loads(result.pop("metrics_json") or "null")
        result["request"] = json.loads(result.pop("request_json"))
        result["failures"] = [dict(f) for f in failures]
        return result

    def list(self):
        with db(self.settings) as connection:
            rows = connection.execute(
                "SELECT * FROM eval_runs ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]
```

```python
# backend/app/api/evaluations.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.dependencies import get_state

router = APIRouter()


class EvaluationRequest(BaseModel):
    variants: list[str]
    item_ids: list[int] | None = None


@router.post("/evaluations", status_code=202)
def start_evaluation(payload: EvaluationRequest, state=Depends(get_state)):
    if state.evaluation is None:
        raise HTTPException(status_code=503, detail="Evaluation service unavailable")
    try:
        run_id = state.evaluation.start(payload.variants, payload.item_ids)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"run_id": run_id, "status": "queued"}


@router.get("/evaluations")
def list_evaluations(state=Depends(get_state)):
    return state.evaluation.list() if state.evaluation else []


@router.get("/evaluations/{run_id}")
def get_evaluation(run_id: str, state=Depends(get_state)):
    result = state.evaluation.get(run_id) if state.evaluation else None
    if result is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return result


@router.get("/evaluations/{run_id}/failures")
def get_failures(
    run_id: str, variant: str | None = None, limit: int = 50,
    state=Depends(get_state),
):
    result = state.evaluation.get(run_id) if state.evaluation else None
    if result is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    failures = result.get("failures", [])
    if variant:
        failures = [row for row in failures if row["variant"] == variant]
    return failures[:limit]
```

Wire:
```python
# dependencies.py build_default_state
from app.services.analysis import AnalysisService
from app.services.captures import CaptureService
from app.services.embeddings import EmbeddingService
from app.services.evaluation import EvaluationService

def build_default_state(settings: Settings) -> AppState:
    from app.ml.registry import ModelRegistry
    from app.pipeline.frame_pipeline import FramePipeline

    registry = ModelRegistry(settings)
    pipeline = FramePipeline(registry, settings)
    embeddings = EmbeddingService(settings, registry)
    return AppState(
        settings,
        registry=registry,
        pipeline=pipeline,
        captures=CaptureService(settings, pipeline),
        analysis=AnalysisService(settings, pipeline, embeddings),
        evaluation=EvaluationService(settings, embeddings),
    )
```
```python
# main.py lifespan
from app.api import evaluations as evaluations_api
from app.services.evaluation import recover_stale_runs

recover_stale_runs(settings)
# in create_app:
app.include_router(evaluations_api.router, prefix="/api")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_evaluation_runner.py tests/test_evaluations_api.py -v`
Expected: runner 1 passed; API tests passed.

- [ ] **Step 5: Commit**

```bash
git add backend/app/evaluation backend/app/services/evaluation.py backend/app/services/captures.py backend/app/api/evaluations.py backend/app/dependencies.py backend/app/main.py backend/app/schemas.py backend/tests/test_evaluation_runner.py backend/tests/test_evaluations_api.py
git commit -m "feat: batch evaluation runner and evaluations API"
```

### Task 17: Meta endpoint, Vite shell, API client, camera, formatting tests

**Files:**
- Create: `backend/app/api/meta.py`, `backend/tests/test_meta.py`
- Create: `frontend/package.json`, `frontend/package-lock.json`, `frontend/vite.config.js`, `frontend/index.html`, `frontend/src/main.js`, `frontend/src/api.js`, `frontend/src/camera.js`, `frontend/src/format.js`, `frontend/src/format.test.js`, `frontend/src/style.css`, `frontend/src/tabs/scan.js`, `frontend/src/tabs/enroll.js`, `frontend/src/tabs/evaluate.js` (temporary no-op stubs replaced in Tasks 18–20)
- Modify: `backend/app/main.py`

**Interfaces:**
- Backend: `GET /api/meta` → `{currency_symbol, enable_clip, variants:[{id,label,crop,embedder}]}`.
- Frontend exports: `formatPrice(cents, symbol)`, `parseVariantIds(text)`, `variantParam(ids)`; `getMeta`, `getItems`, `createItem`, `deleteItem`, `getCaptures`, `uploadCaptures`, `deleteCapture`, `analyzeFrame`, `startEvaluation`, `getEvaluation`, `listEvaluations`; `startCamera(video)`, `stopCamera(stream)`, `captureFrame(video, quality)`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_meta.py
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_meta_lists_variants_and_currency(tmp_path):
    settings = Settings(data_dir=tmp_path, warm_models=False, currency_symbol="$")
    with TestClient(create_app(settings)) as client:
        body = client.get("/api/meta").json()
    assert body["currency_symbol"] == "$"
    assert [v["id"] for v in body["variants"]] == ["v0", "v1", "v2"]
    assert body["variants"][2]["crop"] == "mask"


def test_meta_hides_clip_when_disabled(tmp_path):
    settings = Settings(data_dir=tmp_path, warm_models=False, enable_clip=False)
    with TestClient(create_app(settings)) as client:
        ids = [v["id"] for v in client.get("/api/meta").json()["variants"]]
    assert "v3" not in ids
```

```javascript
// frontend/src/format.test.js
import { describe, expect, it } from "vitest";
import { formatPrice, parseVariantIds, variantParam } from "./format.js";

describe("format helpers", () => {
  it("formats integer cents", () => {
    expect(formatPrice(199, "$")).toBe("$1.99");
    expect(formatPrice(0, "S$")).toBe("S$0.00");
  });

  it("parses and serializes variant ids", () => {
    expect(parseVariantIds(" v0, v2 ,,")).toEqual(["v0", "v2"]);
    expect(variantParam(["v0", "v2"])).toBe("v0,v2");
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run backend: `cd backend && uv run pytest tests/test_meta.py -v` → FAIL (404).
Run frontend after scaffolding (below): `cd frontend && npm install && npm test` → FAIL because helpers are missing.

- [ ] **Step 3: Write minimal implementation**

Backend meta route:

```python
# backend/app/api/meta.py
from fastapi import APIRouter, Request

from app.pipeline.variants import VARIANTS

router = APIRouter()


@router.get("/meta")
def meta(request: Request) -> dict:
    settings = request.app.state.settings
    allowed = {"v0", "v1", "v2"} | ({"v3"} if settings.enable_clip else set())
    return {
        "currency_symbol": settings.currency_symbol,
        "enable_clip": settings.enable_clip,
        "variants": [
            {"id": v.id, "label": v.label, "crop": v.crop, "embedder": v.embedder}
            for key, v in VARIANTS.items()
            if key in allowed
        ],
    }
```

```python
# main.py
from app.api import meta as meta_api

# in create_app:
app.include_router(meta_api.router, prefix="/api")
```

Scaffold the frontend (from repo root):

```bash
npm create vite@latest frontend -- --template vanilla
cd frontend
npm install
npm install -D vitest
rm -f src/counter.js src/javascript.svg public/vite.svg
```

Create temporary tab stubs so the shell builds before Tasks 18–20:

```javascript
// frontend/src/tabs/scan.js
export function initScan() {}
export async function activateScan() {}
export function deactivateScan() {}
```

```javascript
// frontend/src/tabs/enroll.js
export function initEnroll() {}
export async function activateEnroll() {}
export function deactivateEnroll() {}
```

```javascript
// frontend/src/tabs/evaluate.js
export function initEvaluate() {}
export function activateEvaluate() {}
```

Set `frontend/package.json` scripts (keep Vite's versions from the scaffold):

```json
"scripts": {
  "dev": "vite",
  "build": "vite build",
  "preview": "vite preview",
  "test": "vitest run"
}
```

```javascript
// frontend/vite.config.js
import { defineConfig } from "vite";

export default defineConfig({
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/media": "http://127.0.0.1:8000",
    },
  },
});
```

```javascript
// frontend/src/format.js
export function formatPrice(cents, symbol = "$") {
  return `${symbol}${(Number(cents) / 100).toFixed(2)}`;
}

export function parseVariantIds(text) {
  return String(text || "")
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
}

export function variantParam(ids) {
  return ids.join(",");
}
```

```html
<!-- frontend/index.html -->
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Supermarket Auto-Scanner Lab</title>
    <link rel="stylesheet" href="/src/style.css" />
  </head>
  <body>
    <header>
      <h1>Supermarket Auto-Scanner Lab</h1>
      <nav>
        <button data-tab="scan" class="active">Scan</button>
        <button data-tab="enroll">Enroll</button>
        <button data-tab="evaluate">Evaluate</button>
      </nav>
    </header>

    <main>
      <section id="tab-scan" class="tab active">
        <div class="grid two">
          <div>
            <video id="scan-video" autoplay playsinline muted></video>
            <div class="row">
              <button id="scan-analyze">Analyze now</button>
              <label><input id="scan-auto" type="checkbox" /> Auto (~1 fps)</label>
            </div>
            <div id="scan-variants" class="chips"></div>
            <div id="scan-status" class="status"></div>
          </div>
          <div>
            <img id="scan-annotated" alt="Annotated frame" />
            <div id="scan-results"></div>
            <pre id="scan-timings"></pre>
          </div>
        </div>
      </section>

      <section id="tab-enroll" class="tab">
        <form id="item-form" class="row">
          <input id="item-name" placeholder="Item name" required maxlength="120" />
          <input id="item-price" type="number" min="0" step="0.01" placeholder="Price" required />
          <button type="submit">Add item</button>
        </form>
        <div id="item-list" class="list"></div>
        <div id="enroll-panel" hidden>
          <h2 id="selected-item-name"></h2>
          <div class="grid two">
            <div>
              <video id="enroll-video" autoplay playsinline muted></video>
              <div class="row">
                <label>Session <input id="enroll-session" size="10" /></label>
                <button id="enroll-new-session" type="button">New session</button>
              </div>
              <div class="row">
                <button type="button" data-role="gallery">Capture gallery</button>
                <button type="button" data-role="test">Capture test</button>
                <button id="enroll-delete-item" type="button" class="danger">Delete item</button>
              </div>
              <div id="capture-status" class="status"></div>
            </div>
            <div>
              <div id="capture-counts" class="muted"></div>
              <div id="capture-grid" class="capture-grid"></div>
            </div>
          </div>
        </div>
      </section>

      <section id="tab-evaluate" class="tab">
        <div class="row">
          <button id="eval-run">Run evaluation</button>
          <span class="muted">Batch only — scores held-out test captures.</span>
        </div>
        <div id="eval-variants" class="chips"></div>
        <div id="eval-status" class="status"></div>
        <div id="eval-results"></div>
        <div id="eval-failures"></div>
      </section>
    </main>

    <script type="module" src="/src/main.js"></script>
  </body>
</html>
```

```javascript
// frontend/src/api.js
async function jsonFetch(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = await response.text();
    try {
      detail = JSON.parse(detail).detail || detail;
    } catch {
      /* keep text */
    }
    throw new Error(`${response.status}: ${detail}`);
  }
  return response.json();
}

export const getMeta = () => jsonFetch("/api/meta");
export const getHealth = () => jsonFetch("/api/health");
export const getItems = () => jsonFetch("/api/items");

export const createItem = (name, priceCents) =>
  jsonFetch("/api/items", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, price_cents: priceCents }),
  });

export const deleteItem = (itemId) =>
  fetch(`/api/items/${itemId}`, { method: "DELETE" });

export const getCaptures = (itemId, role) =>
  jsonFetch(`/api/items/${itemId}/captures${role ? `?role=${role}` : ""}`);

export async function uploadCaptures(itemId, role, sessionId, blobs) {
  const form = new FormData();
  for (const blob of blobs) form.append("files", blob, "frame.jpg");
  return jsonFetch(
    `/api/items/${itemId}/captures?role=${role}&session_id=${encodeURIComponent(sessionId)}`,
    { method: "POST", body: form },
  );
}

export const deleteCapture = (itemId, captureId) =>
  fetch(`/api/items/${itemId}/captures/${captureId}`, { method: "DELETE" });

export async function analyzeFrame(blob, variants) {
  const form = new FormData();
  form.append("image", blob, "frame.jpg");
  form.append("variants", variants.join(","));
  return jsonFetch("/api/analyze", { method: "POST", body: form });
}

export const startEvaluation = (variants) =>
  jsonFetch("/api/evaluations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ variants }),
  });

export const getEvaluation = (runId) => jsonFetch(`/api/evaluations/${runId}`);
export const listEvaluations = () => jsonFetch("/api/evaluations");
```

```javascript
// frontend/src/camera.js
export async function startCamera(video, streamRef = {}) {
  if (streamRef.stream) return streamRef.stream;
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
    audio: false,
  });
  streamRef.stream = stream;
  video.srcObject = stream;
  await video.play();
  return stream;
}

export function stopCamera(stream) {
  if (stream) stream.getTracks().forEach((track) => track.stop());
}

export function captureFrame(video, quality = 0.9) {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("Could not encode frame"))),
      "image/jpeg",
      quality,
    );
  });
}
```

```javascript
// frontend/src/main.js
import { getMeta } from "./api.js";
import { initScan, activateScan, deactivateScan } from "./tabs/scan.js";
import { initEnroll, activateEnroll, deactivateEnroll } from "./tabs/enroll.js";
import { initEvaluate, activateEvaluate } from "./tabs/evaluate.js";

const tabs = {
  scan: { activate: activateScan, deactivate: deactivateScan },
  enroll: { activate: activateEnroll, deactivate: deactivateEnroll },
  evaluate: { activate: activateEvaluate, deactivate: () => {} },
};
let activeTab = "scan";

function showTab(name) {
  if (name === activeTab) return;
  tabs[activeTab]?.deactivate?.();
  document.querySelectorAll(".tab").forEach((el) => el.classList.remove("active"));
  document.querySelectorAll("nav button").forEach((el) => {
    el.classList.toggle("active", el.dataset.tab === name);
  });
  document.getElementById(`tab-${name}`).classList.add("active");
  activeTab = name;
  tabs[name]?.activate?.().catch((error) => {
    const status = document.getElementById(`${name}-status`);
    if (status) status.textContent = error.message;
  });
}

async function boot() {
  document.querySelectorAll("nav button").forEach((button) => {
    button.addEventListener("click", () => showTab(button.dataset.tab));
  });
  const meta = await getMeta();
  initScan(meta);
  initEnroll(meta);
  initEvaluate(meta);
  await activateScan();
}

boot().catch((error) => {
  document.body.insertAdjacentHTML(
    "afterbegin",
    `<p class="status error">Startup error: ${error.message}</p>`,
  );
});
```

```css
/* frontend/src/style.css */
:root { color-scheme: light dark; font-family: ui-sans-serif, system-ui, sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; background: #0f1115; color: #e8eaef; }
header { display: flex; align-items: center; justify-content: space-between; padding: 12px 20px; border-bottom: 1px solid #2a2f3a; }
h1 { font-size: 18px; margin: 0; }
nav button { background: transparent; border: 1px solid #3a4150; color: inherit; padding: 6px 12px; border-radius: 8px; margin-left: 6px; cursor: pointer; }
nav button.active { background: #2563eb; border-color: #2563eb; }
main { padding: 20px; }
.tab { display: none; }
.tab.active { display: block; }
.grid.two { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
video, #scan-annotated { width: 100%; max-height: 420px; background: #000; border-radius: 12px; object-fit: contain; }
#scan-annotated { min-height: 240px; }
.row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin: 10px 0; }
button { background: #1f2937; color: inherit; border: 1px solid #3a4150; padding: 8px 12px; border-radius: 8px; cursor: pointer; }
button.primary { background: #2563eb; border-color: #2563eb; }
button.danger { background: #7f1d1d; border-color: #b91c1c; }
input { background: #151922; color: inherit; border: 1px solid #3a4150; padding: 8px; border-radius: 8px; }
.chips { display: flex; gap: 8px; flex-wrap: wrap; }
.chips label { border: 1px solid #3a4150; padding: 6px 10px; border-radius: 999px; }
.status { min-height: 24px; margin: 10px 0; color: #93c5fd; }
.status.error { color: #fca5a5; }
.muted { color: #94a3b8; font-size: 13px; }
.card { border: 1px solid #2a2f3a; border-radius: 12px; padding: 12px; margin: 8px 0; background: #151922; }
.card strong { font-size: 18px; }
.list button { display: block; width: 100%; text-align: left; margin: 6px 0; }
.capture-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(110px, 1fr)); gap: 8px; }
.capture-grid figure { margin: 0; }
.capture-grid img { width: 100%; border-radius: 8px; background: #000; }
.capture-grid figcaption { font-size: 11px; color: #94a3b8; }
pre { background: #151922; padding: 12px; border-radius: 8px; overflow: auto; font-size: 12px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }
th, td { border: 1px solid #2a2f3a; padding: 6px 8px; text-align: left; }
@media (max-width: 900px) { .grid.two { grid-template-columns: 1fr; } }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && uv run pytest tests/test_meta.py -v` → 2 passed.
Run: `cd frontend && npm test` → 1 test file passed; `npm run build` succeeds.

- [ ] **Step 5: Commit**

```bash
git add backend/app/api/meta.py backend/tests/test_meta.py backend/app/main.py frontend
git commit -m "feat: meta endpoint and vite app shell"
```

---

### Task 18: Scan tab

**Files:**
- Create: `frontend/src/tabs/scan.js`, `frontend/src/tabs/scan_logic.js`, `frontend/src/tabs/scan_logic.test.js`
- Modify: `frontend/src/main.js` only if imports differ (they are already wired in Task 17).

**Interfaces:**
- `initScan(meta)`, `activateScan()`, `deactivateScan()`.
- Pure helpers `bestTop(variants)`, `statusMessage(body)`.

- [ ] **Step 1: Write the failing test**

```javascript
// frontend/src/tabs/scan_logic.test.js
import { describe, expect, it } from "vitest";
import { bestTop, statusMessage } from "./scan_logic.js";

describe("scan logic", () => {
  it("picks the highest-scoring top-1 across variants", () => {
    const best = bestTop([
      { top_items: [{ name: "Milk", score: 0.7 }] },
      { top_items: [{ name: "Bread", score: 0.9 }] },
    ]);
    expect(best.name).toBe("Bread");
  });

  it("explains no-hand and rejected-mask states", () => {
    expect(statusMessage({ hand_found: false, mask_found: false })).toMatch(/No hand/);
    expect(statusMessage({ hand_found: true, mask_found: false })).toMatch(/rejected/);
    expect(statusMessage({ hand_found: true, mask_found: true })).toBe("OK");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test`
Expected: FAIL because `scan_logic.js` is missing.

- [ ] **Step 3: Write minimal implementation**

```javascript
// frontend/src/tabs/scan_logic.js
export function bestTop(variants) {
  return (
    variants
      .flatMap((variant) => variant.top_items || [])
      .sort((a, b) => b.score - a.score)[0] || null
  );
}

export function statusMessage(body) {
  if (!body.hand_found) return "No hand detected — hold the item up to the camera.";
  if (!body.mask_found) return "Hand found, but SAM2 rejected the mask.";
  return "OK";
}
```

```javascript
// frontend/src/tabs/scan.js
import { analyzeFrame } from "../api.js";
import { captureFrame, startCamera, stopCamera } from "../camera.js";
import { formatPrice } from "../format.js";
import { bestTop, statusMessage } from "./scan_logic.js";

let meta = null;
let stream = null;
let streamRef = {};
let timer = null;
let busy = false;
let selected = new Set();

function renderVariants() {
  const container = document.getElementById("scan-variants");
  container.replaceChildren();
  selected = new Set();
  for (const variant of meta.variants) {
    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = variant.id;
    checkbox.checked = variant.id !== "v3";
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) selected.add(variant.id);
      else selected.delete(variant.id);
    });
    if (checkbox.checked) selected.add(variant.id);
    label.append(checkbox, ` ${variant.label}`);
    container.append(label);
  }
}

function renderResult(body) {
  document.getElementById("scan-annotated").src =
    `data:image/jpeg;base64,${body.annotated_jpeg_b64}`;
  document.getElementById("scan-status").textContent = statusMessage(body);
  document.getElementById("scan-status").classList.toggle(
    "error",
    !body.hand_found || !body.mask_found,
  );

  const results = document.getElementById("scan-results");
  results.replaceChildren();
  for (const variant of body.variants) {
    const card = document.createElement("div");
    card.className = "card";
    const title = document.createElement("div");
    title.innerHTML = `<strong>${
      meta.variants.find((v) => v.id === variant.variant)?.label || variant.variant
    }</strong> <span class="muted">${variant.status}</span>`;
    card.append(title);
    if (variant.status === "ok" && variant.top_items.length) {
      for (const [index, item] of variant.top_items.entries()) {
        const row = document.createElement("div");
        row.textContent = `${index + 1}. ${item.name} — ${formatPrice(
          item.price_cents,
          meta.currency_symbol,
        )} (${item.score.toFixed(3)})`;
        card.append(row);
      }
    } else {
      const row = document.createElement("div");
      row.className = "muted";
      row.textContent =
        variant.status === "no_gallery"
          ? "No gallery captures yet — enroll items first."
          : variant.status;
      card.append(row);
    }
    results.append(card);
  }

  document.getElementById("scan-timings").textContent = JSON.stringify(
    body.variants.map((v) => ({ variant: v.variant, timings_ms: v.timings_ms })),
    null,
    2,
  );
}

async function analyzeOnce() {
  if (busy) return;
  if (!selected.size) {
    document.getElementById("scan-status").textContent = "Select at least one variant.";
    return;
  }
  const video = document.getElementById("scan-video");
  if (!video.videoWidth) return;
  busy = true;
  document.getElementById("scan-status").textContent = "Analyzing…";
  try {
    const blob = await captureFrame(video);
    const body = await analyzeFrame(blob, [...selected]);
    renderResult(body);
  } catch (error) {
    document.getElementById("scan-status").textContent = error.message;
    document.getElementById("scan-status").classList.add("error");
  } finally {
    busy = false;
  }
}

function setAuto(enabled) {
  if (timer) clearInterval(timer);
  timer = enabled ? setInterval(analyzeOnce, 1100) : null;
}

export function initScan(metaData) {
  meta = metaData;
  renderVariants();
  document.getElementById("scan-analyze").addEventListener("click", analyzeOnce);
  document.getElementById("scan-auto").addEventListener("change", (event) => {
    setAuto(event.target.checked);
  });
}

export async function activateScan() {
  if (!stream) {
    stream = await startCamera(
      document.getElementById("scan-video"),
      streamRef,
    );
  }
}

export function deactivateScan() {
  document.getElementById("scan-auto").checked = false;
  setAuto(false);
  stopCamera(stream);
  stream = null;
  streamRef = {};
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test && npm run build` → pass.
Manual: with the backend running, open the Scan tab; camera preview appears; Analyze returns cards or a `no_gallery` hint.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/tabs/scan.js frontend/src/tabs/scan_logic.js frontend/src/tabs/scan_logic.test.js
git commit -m "feat: live scan tab"
```

---

### Task 19: Enroll tab

**Files:**
- Create: `frontend/src/tabs/enroll.js`
- Modify: none (already imported by `main.js`).

**Interfaces:**
- `initEnroll(meta)`, `activateEnroll()`, `deactivateEnroll()`.

- [ ] **Step 1: Write the failing test**

The enroll flow is DOM- and camera-driven; verify with a manual test after implementation (no useful pure function to unit-test beyond formatting, already covered). Capture this checklist as the failing acceptance check:

1. Create an item → it appears in the item list.
2. Select it → enroll panel and camera appear.
3. Capture gallery / test → backend returns accepted or a rejection reason.
4. Captures appear as thumbnails with counts; delete works.

- [ ] **Step 2: Confirm the acceptance check fails**

Run the app and attempt the checklist. Expected: the Enroll tab is empty because `enroll.js` does not exist.

- [ ] **Step 3: Write minimal implementation**

```javascript
// frontend/src/tabs/enroll.js
import {
  createItem,
  deleteCapture,
  deleteItem,
  getCaptures,
  getItems,
  uploadCaptures,
} from "../api.js";
import { captureFrame, startCamera, stopCamera } from "../camera.js";
import { formatPrice } from "../format.js";

let meta = null;
let selectedItem = null;
let stream = null;
let streamRef = {};
let sessionId = crypto.randomUUID().slice(0, 8);

function itemButton(item) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = `${item.name} — ${formatPrice(
    item.price_cents,
    meta.currency_symbol,
  )} (gallery ${item.gallery_count} / test ${item.test_count})`;
  button.addEventListener("click", () => selectItem(item));
  return button;
}

async function refreshItems() {
  const items = await getItems();
  const list = document.getElementById("item-list");
  list.replaceChildren();
  for (const item of items) list.append(itemButton(item));
  if (selectedItem) {
    const updated = items.find((item) => item.id === selectedItem.id);
    if (!updated) clearSelection();
    else selectedItem = updated;
  }
}

function clearSelection() {
  selectedItem = null;
  document.getElementById("enroll-panel").hidden = true;
  document.getElementById("capture-grid").replaceChildren();
  deactivateEnroll();
}

async function selectItem(item) {
  selectedItem = item;
  document.getElementById("enroll-panel").hidden = false;
  document.getElementById("selected-item-name").textContent = item.name;
  document.getElementById("enroll-session").value = sessionId;
  await activateEnroll();
  await refreshCaptures();
}

async function refreshCaptures() {
  if (!selectedItem) return;
  const [gallery, test] = await Promise.all([
    getCaptures(selectedItem.id, "gallery"),
    getCaptures(selectedItem.id, "test"),
  ]);
  document.getElementById("capture-counts").textContent =
    `Gallery: ${gallery.length} · Test: ${test.length}`;
  const grid = document.getElementById("capture-grid");
  grid.replaceChildren();
  for (const capture of [...gallery, ...test]) {
    const figure = document.createElement("figure");
    const image = document.createElement("img");
    image.src = `/media/${capture.thumb_rel_path}`;
    image.alt = `${capture.role} capture`;
    const caption = document.createElement("figcaption");
    caption.textContent = `${capture.role} · ${capture.mask_status}`;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "Delete";
    remove.className = "danger";
    remove.addEventListener("click", async () => {
      await deleteCapture(selectedItem.id, capture.id);
      await refreshCaptures();
    });
    figure.append(image, caption, remove);
    grid.append(figure);
  }
}

async function capture(role, button) {
  if (!selectedItem) return;
  button.disabled = true;
  const status = document.getElementById("capture-status");
  status.textContent = "Capturing and segmenting…";
  try {
    const blob = await captureFrame(document.getElementById("enroll-video"));
    const result = await uploadCaptures(
      selectedItem.id,
      role,
      document.getElementById("enroll-session").value || sessionId,
      [blob],
    );
    if (result.accepted.length) {
      status.textContent = `Accepted ${role} capture.`;
    } else {
      status.textContent = `Rejected: ${result.rejected[0]?.reason || "unknown"} — retake.`;
    }
    await refreshItems();
    await refreshCaptures();
  } catch (error) {
    status.textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

export function initEnroll(metaData) {
  meta = metaData;
  document.getElementById("item-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const name = document.getElementById("item-name").value.trim();
    const price = Number(document.getElementById("item-price").value);
    if (!name || Number.isNaN(price) || price < 0) return;
    await createItem(name, Math.round(price * 100));
    event.target.reset();
    await refreshItems();
  });

  document.getElementById("enroll-new-session").addEventListener("click", () => {
    sessionId = crypto.randomUUID().slice(0, 8);
    document.getElementById("enroll-session").value = sessionId;
  });

  document.querySelectorAll("#enroll-panel [data-role]").forEach((button) => {
    button.addEventListener("click", () => capture(button.dataset.role, button));
  });

  document.getElementById("enroll-delete-item").addEventListener("click", async () => {
    if (!selectedItem) return;
    if (!confirm(`Delete ${selectedItem.name} and all its captures?`)) return;
    await deleteItem(selectedItem.id);
    clearSelection();
    await refreshItems();
  });

  refreshItems().catch((error) => {
    document.getElementById("capture-status").textContent = error.message;
  });
}

export async function activateEnroll() {
  if (!selectedItem) return;
  if (!stream) {
    stream = await startCamera(
      document.getElementById("enroll-video"),
      streamRef,
    );
  }
}

export function deactivateEnroll() {
  stopCamera(stream);
  stream = null;
  streamRef = {};
}
```

- [ ] **Step 4: Verify manually**

Run backend and frontend; complete the acceptance checklist above. Confirm rejected captures show the reason and are not stored.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/tabs/enroll.js
git commit -m "feat: enrollment tab with gallery and test captures"
```

---

### Task 20: Evaluate tab

**Files:**
- Create: `frontend/src/tabs/evaluate.js`
- Modify: none (already imported by `main.js`).

**Interfaces:**
- `initEvaluate(meta)`, `activateEvaluate()`.

- [ ] **Step 1: Acceptance check (fails initially)**

With at least 2 items having gallery and test captures:
1. Select variants and click Run evaluation.
2. A progress count updates.
3. Metrics table shows top-1/top-5/macro/CI/n, plus hand-found %, mask-accepted %, near-duplicate count, and an insufficient-data note when there are fewer than 5 items.
4. Failure entries show true/predicted and link to rendered frames.

Expected before implementation: the Evaluate tab does nothing because `evaluate.js` is missing.

- [ ] **Step 2: Confirm the acceptance check fails**

Run the app and attempt the checklist. Expected: no run starts.

- [ ] **Step 3: Write minimal implementation**

```javascript
// frontend/src/tabs/evaluate.js
import { getEvaluation, listEvaluations, startEvaluation } from "../api.js";

let meta = null;
let selected = new Set();
let polling = null;

function renderVariants() {
  const container = document.getElementById("eval-variants");
  container.replaceChildren();
  selected = new Set();
  for (const variant of meta.variants) {
    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = variant.id;
    checkbox.checked = variant.id !== "v3";
    if (checkbox.checked) selected.add(variant.id);
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) selected.add(variant.id);
      else selected.delete(variant.id);
    });
    label.append(checkbox, ` ${variant.label}`);
    container.append(label);
  }
}

function renderMetrics(run) {
  const container = document.getElementById("eval-results");
  container.replaceChildren();
  if (!run.metrics) {
    container.textContent = "No metrics yet.";
    return;
  }
  const table = document.createElement("table");
  table.innerHTML = `
    <thead><tr>
      <th>Variant</th><th>N</th><th>Top-1</th><th>Top-1 95% CI</th>
      <th>Top-5</th><th>Macro top-1</th><th>p50 total ms</th><th>p95 total ms</th>
    </tr></thead>`;
  const body = document.createElement("tbody");
  for (const [variant, metrics] of Object.entries(run.metrics)) {
    const row = document.createElement("tr");
    const ci = metrics.top1_ci
      ? `${(metrics.top1_ci[0] * 100).toFixed(0)}–${(metrics.top1_ci[1] * 100).toFixed(0)}%`
      : "—";
    const total = metrics.latency_ms?.total || {};
    row.innerHTML = `
      <td>${variant}</td>
      <td>${metrics.n}</td>
      <td>${(metrics.top1 * 100).toFixed(1)}%</td>
      <td>${ci}</td>
      <td>${(metrics.top5 * 100).toFixed(1)}%</td>
      <td>${(metrics.macro_top1 * 100).toFixed(1)}%</td>
      <td>${(total.p50 || 0).toFixed(0)}</td>
      <td>${(total.p95 || 0).toFixed(0)}</td>`;
    body.append(row);
  }
  table.append(body);
  container.append(table);

  const first = Object.values(run.metrics)[0];
  const kNote = document.createElement("p");
  kNote.className = "muted";
  kNote.textContent = `Top-5 effective k: ${first?.k_effective ?? "—"} (warning when fewer than 5 items).`;
  container.append(kNote);

  if (first) {
    const dataNote = document.createElement("p");
    dataNote.className = "muted";
    dataNote.textContent =
      `Items: ${first.n_items} · Hand found: ${(first.hand_found_rate * 100).toFixed(0)}% · ` +
      `Mask accepted: ${(first.mask_accepted_rate * 100).toFixed(0)}% · ` +
      `Near-duplicate test images: ${first.near_duplicate_count}` +
      (first.n_items < 5 ? " — insufficient data: fewer than 5 items, top-5 is not meaningful." : "");
    container.append(dataNote);
  }
}

function renderFailures(run) {
  const container = document.getElementById("eval-failures");
  container.replaceChildren();
  if (!run.failures?.length) {
    container.textContent = "No failures recorded.";
    return;
  }
  const heading = document.createElement("h3");
  heading.textContent = "Failures";
  container.append(heading);
  for (const failure of run.failures) {
    const card = document.createElement("div");
    card.className = "card";
    const link = document.createElement("a");
    link.href = `/media/${failure.rendered_rel_path}`;
    link.target = "_blank";
    link.textContent = "View frame";
    card.innerHTML = `<strong>${failure.true_item_name}</strong> → ${
      failure.pred_item_name || "none"
    } <span class="muted">(${failure.variant}, ${failure.sample_status})</span> `;
    card.append(link);
    container.append(card);
  }
}

async function poll(runId) {
  if (polling) clearInterval(polling);
  polling = setInterval(async () => {
    try {
      const run = await getEvaluation(runId);
      document.getElementById("eval-status").textContent =
        `${run.status} · ${run.progress_done}/${run.progress_total}`;
      if (run.status === "completed" || run.status === "failed") {
        clearInterval(polling);
        polling = null;
        renderMetrics(run);
        renderFailures(run);
        if (run.error) {
          document.getElementById("eval-status").textContent = `Failed: ${run.error}`;
        }
      }
    } catch (error) {
      clearInterval(polling);
      polling = null;
      document.getElementById("eval-status").textContent = error.message;
    }
  }, 1000);
}

async function loadLatest() {
  try {
    const runs = await listEvaluations();
    if (!runs.length) return;
    const run = await getEvaluation(runs[0].id);
    renderMetrics(run);
    renderFailures(run);
    document.getElementById("eval-status").textContent = `${run.status} · ${runs[0].id}`;
  } catch (error) {
    document.getElementById("eval-status").textContent = error.message;
  }
}

export function initEvaluate(metaData) {
  meta = metaData;
  renderVariants();
  document.getElementById("eval-run").addEventListener("click", async () => {
    if (!selected.size) {
      document.getElementById("eval-status").textContent = "Select at least one variant.";
      return;
    }
    document.getElementById("eval-status").textContent = "Starting…";
    try {
      const result = await startEvaluation([...selected]);
      document.getElementById("eval-status").textContent = `Queued ${result.run_id}`;
      poll(result.run_id);
    } catch (error) {
      document.getElementById("eval-status").textContent = error.message;
    }
  });
  loadLatest();
}

export function activateEvaluate() {
  // No camera needed; polling continues in the background.
}
```

- [ ] **Step 4: Verify manually**

Run the acceptance checklist. Confirm the metrics table, progress, and failure links. Confirm a run with fewer than 5 items still completes and shows the `k_effective` note.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/tabs/evaluate.js
git commit -m "feat: batch evaluation tab"
```

---

### Task 21: Model download, smoke script, README, environment files

**Files:**
- Create: `scripts/download_models.py`, `scripts/smoke_models.py`, `scripts/run_backend.sh`, `scripts/run_frontend.sh`, `.env.example`
- Modify: `README.md`, `.gitignore`

**Interfaces:**
- `download_models.py [--with-clip]` downloads HF snapshots to `data/hf_cache` and the MediaPipe task to `data/assets/hand_landmarker.task`, writing `data/model_manifest.json`.
- `smoke_models.py` loads the models on the resolved device and prints dimensions/timings.
- Run scripts start the backend on `127.0.0.1:8000` and the Vite dev server.

- [ ] **Step 1: Write the failing check**

```bash
cd backend && uv run python ../scripts/download_models.py --help
cd backend && uv run python ../scripts/smoke_models.py --help
```
Expected: FAIL because the scripts do not exist.

- [ ] **Step 2: Confirm the failure**

Run the commands above. Expected: `No such file or directory`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/download_models.py
import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("AUTO_SCANNER_DATA_DIR", str(ROOT / "data")))
os.environ.setdefault("HF_HOME", str(DATA / "hf_cache"))

from huggingface_hub import HfApi, snapshot_download  # noqa: E402

HAND_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

REPOS = {
    "sam2": "facebook/sam2.1-hiera-tiny",
    "dinov2": "facebook/dinov2-small",
    "clip": "openai/clip-vit-base-patch32",
}


def download_repo(repo_id: str) -> dict:
    revision = HfApi().model_info(repo_id).sha
    hub_cache = Path(os.environ["HF_HOME"]) / "hub"
    path = snapshot_download(
        repo_id=repo_id, revision=revision, cache_dir=str(hub_cache)
    )
    return {"repo_id": repo_id, "revision": revision, "cache_path": path}


def download_hand() -> dict:
    destination = DATA / "assets" / "hand_landmarker.task"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        urllib.request.urlretrieve(HAND_URL, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return {"url": HAND_URL, "path": str(destination), "sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-clip", action="store_true", help="Also download CLIP")
    args = parser.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    keys = ["sam2", "dinov2"] + (["clip"] if args.with_clip else [])
    manifest = {
        "models": {key: download_repo(REPOS[key]) for key in keys},
        "hand": download_hand(),
    }
    (DATA / "model_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
```

```python
# scripts/smoke_models.py
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import bootstrap  # noqa: E402,F401
from app.config import Settings  # noqa: E402
from app.ml.devices import resolve_device  # noqa: E402
from app.pipeline.embedders import ClipEmbedder, DinoV2Embedder  # noqa: E402
from app.pipeline.hand import HandDetector  # noqa: E402
from app.pipeline.sam2 import Sam2Segmenter  # noqa: E402
from app.pipeline.types import PromptSpec  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-clip", action="store_true")
    args = parser.parse_args()

    settings = Settings()
    device = resolve_device(settings.device)
    image = np.full((480, 640, 3), 127, dtype=np.uint8)
    print(f"device={device}")

    if settings.hand_asset.exists():
        started = time.perf_counter()
        detector = HandDetector(settings.hand_asset)
        result = detector.detect(image)
        print(f"hand: loaded, detect(non-hand)={result}, {(time.perf_counter() - started) * 1000:.0f} ms")
    else:
        print(f"hand: asset missing at {settings.hand_asset}")

    started = time.perf_counter()
    dino = DinoV2Embedder(settings.dinov2_model_id, device)
    vector = dino.embed(image)
    print(f"dinov2: dim={vector.size} norm={float(np.linalg.norm(vector)):.4f} "
          f"{(time.perf_counter() - started) * 1000:.0f} ms")

    started = time.perf_counter()
    sam2 = Sam2Segmenter(settings.sam2_model_id, device)
    prompt = PromptSpec([(320.0, 240.0)], [], (200.0, 150.0, 440.0, 330.0), "pinch_roi")
    masks = sam2.segment(image, prompt)
    print(f"sam2: candidates={len(masks)} shapes={[m.mask.shape for m in masks]} "
          f"{(time.perf_counter() - started) * 1000:.0f} ms")

    if args.with_clip:
        started = time.perf_counter()
        clip = ClipEmbedder(settings.clip_model_id, device)
        vector = clip.embed(image)
        print(f"clip: dim={vector.size} norm={float(np.linalg.norm(vector)):.4f} "
              f"{(time.perf_counter() - started) * 1000:.0f} ms")


if __name__ == "__main__":
    main()
```

```bash
# scripts/run_backend.sh
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../backend"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$(pwd)/../data/hf_cache}"
exec uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 "$@"
```

```bash
# scripts/run_frontend.sh
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../frontend"
exec npm run dev
```

```text
# .env.example
AUTO_SCANNER_DEVICE=auto
AUTO_SCANNER_CURRENCY_SYMBOL=$
AUTO_SCANNER_WARM_MODELS=true
AUTO_SCANNER_ENABLE_CLIP=false
PYTORCH_ENABLE_MPS_FALLBACK=1
HF_HOME=./data/hf_cache
TOKENIZERS_PARALLELISM=false
```

````markdown
<!-- README.md -->
# Supermarket Auto-Scanner — Model Validation Lab

Local prototype for recognizing a supermarket item held in front of a laptop
camera and retrieving its name and price. It exists to answer: are hand-aware
SAM2 segmentation and DINOv2 retrieval accurate enough for a final scanner?

## Stack

- FastAPI + SQLite + local files; all inference server-side.
- MediaPipe HandLandmarker → SAM2 (`facebook/sam2.1-hiera-tiny`) → DINOv2
  (`facebook/dinov2-small`) or CLIP (`openai/clip-vit-base-patch32`).
- Vite vanilla JavaScript UI: **Scan**, **Enroll**, **Evaluate**.
- Python 3.12 via `uv`; Apple M1 with MPS, CPU fallback.

## Setup

```bash
cd backend
uv sync --python 3.12
uv run python ../scripts/download_models.py          # add --with-clip for V3
cd ../frontend
npm ci
```

## Run

Terminal 1:

```bash
scripts/run_backend.sh
```

Terminal 2:

```bash
scripts/run_frontend.sh
```

Open http://127.0.0.1:5173. `GET http://127.0.0.1:8000/api/health` shows model
and device status.

## Evaluation protocol

1. **Enroll** each item with 3+ gallery captures from multiple angles, all
   held in a hand, in session `G1`.
2. Capture 5–10 **test** images in a *different* session `T1` with different
   angles/lighting/distance. Never put test images in the gallery.
3. Run **Evaluate** with V0/V1/V2 (and V3 if CLIP is downloaded).
4. Read top-1/top-5, the 95% confidence interval, per-item results, and the
   failure gallery. V0/V1 are controls: if V0 ≫ V2, suspect background
   shortcutting. Tune mask thresholds only on separate debug frames.

Minimum for a weak signal: 5 items × 3 gallery × 5 test. For a credible
go/no-go: 8–12 visually similar items × 10–20 test captures each.

## Tests

```bash
cd backend && uv run pytest -v
cd frontend && npm test && npm run build
cd backend && RUN_MODEL_TESTS=1 uv run pytest -m slow -v
```

## Notes

- Evaluation is batch/offline; Scan is live inference and reports no accuracy.
- SAM2 produces masks, not embeddings. DINOv2 is the primary embedder; CLIP is
  a semantic baseline.
- Matching is exact cosine. HNSW is deferred behind `VectorIndex`.
- DINOv2 weight licensing must be verified before any commercial use.
````

Append to `.gitignore`:

```gitignore
# Project data and local tooling
data/
backend/assets/*.task
backend/.venv/
.env
node_modules/
frontend/node_modules/
frontend/dist/
frontend/.vite/
__pycache__/
*.py[cod]
.DS_Store
```

- [ ] **Step 4: Verify**

Run:
```bash
cd backend && uv run python ../scripts/download_models.py --help
cd backend && uv run python ../scripts/smoke_models.py --help
chmod +x ../scripts/run_backend.sh ../scripts/run_frontend.sh
```
Expected: both print usage and exit 0. (Full downloads happen in Task 22.)

- [ ] **Step 5: Commit**

```bash
git add scripts .env.example README.md .gitignore
git commit -m "chore: model download, smoke scripts, and setup docs"
```

---

### Task 22: Slow model tests, hardening, and end-to-end verification

**Files:**
- Create: `backend/tests/test_models_slow.py`
- Modify: `backend/app/api/analyze.py`, `backend/app/main.py` if hardening changes are needed
- Test: the full suite

**Interfaces:**
- Slow tests are gated by `RUN_MODEL_TESTS=1` and `@pytest.mark.slow`.

- [ ] **Step 1: Write the failing slow test**

```python
# backend/tests/test_models_slow.py
import os

import numpy as np
import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.getenv("RUN_MODEL_TESTS") != "1",
        reason="set RUN_MODEL_TESTS=1 with downloaded model assets",
    ),
]


def test_dinov2_embedding_contract():
    from app.config import Settings
    from app.ml.devices import resolve_device
    from app.pipeline.embedders import DinoV2Embedder

    settings = Settings()
    embedder = DinoV2Embedder(settings.dinov2_model_id, resolve_device(settings.device))
    vector = embedder.embed(np.full((224, 224, 3), 127, dtype=np.uint8))
    assert vector.shape == (384,)
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-3)


def test_sam2_returns_three_deterministic_masks():
    from app.config import Settings
    from app.ml.devices import resolve_device
    from app.pipeline.sam2 import Sam2Segmenter
    from app.pipeline.types import PromptSpec

    settings = Settings()
    sam2 = Sam2Segmenter(settings.sam2_model_id, resolve_device(settings.device))
    image = np.full((480, 640, 3), 127, dtype=np.uint8)
    prompt = PromptSpec([(320.0, 240.0)], [], (200.0, 150.0, 440.0, 330.0), "pinch_roi")
    first = sam2.segment(image, prompt)
    second = sam2.segment(image, prompt)
    assert len(first) == len(second) == 3
    for a, b in zip(first, second):
        union = np.logical_or(a.mask, b.mask).sum()
        intersection = np.logical_and(a.mask, b.mask).sum()
        iou = intersection / union if union else 1.0
        assert iou >= 0.9


def test_hand_asset_loads_and_handles_blank_frame():
    from app.config import Settings
    from app.pipeline.hand import HandDetector

    settings = Settings()
    if not settings.hand_asset.exists():
        pytest.skip("hand_landmarker.task not downloaded")
    detector = HandDetector(settings.hand_asset)
    assert detector.detect(np.zeros((480, 640, 3), dtype=np.uint8)) is None


def test_clip_embedding_contract_when_enabled():
    import os as _os

    if _os.getenv("AUTO_SCANNER_ENABLE_CLIP") != "1":
        pytest.skip("CLIP disabled")
    from app.config import Settings
    from app.ml.devices import resolve_device
    from app.pipeline.embedders import ClipEmbedder

    settings = Settings(enable_clip=True)
    embedder = ClipEmbedder(settings.clip_model_id, resolve_device(settings.device))
    vector = embedder.embed(np.full((224, 224, 3), 127, dtype=np.uint8))
    assert vector.shape == (512,)
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-3)
```

- [ ] **Step 2: Run the slow test (after downloading models)**

Run: `cd backend && RUN_MODEL_TESTS=1 uv run pytest -m slow -v`
Expected: passes, or skips the CLIP/hand cases if those assets are absent. If SAM2 post-processing shape differs from the wrapper assumption, fix `pipeline/sam2.py` until the three-mask/determinism test passes.

- [ ] **Step 3: Add the hardening checks**

```python
# backend/tests/test_hardening.py
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import db, init_db
from app.main import create_app
from app.services.evaluation import recover_stale_runs


def test_analyze_rejects_oversized_upload(tmp_path):
    settings = Settings(data_dir=tmp_path, warm_models=False, max_upload_bytes=10)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/analyze",
            files={"image": ("frame.jpg", b"x" * 20, "image/jpeg")},
            data={"variants": "v0"},
        )
    assert response.status_code == 413


def test_stale_runs_are_failed_on_restart(tmp_path):
    settings = Settings(data_dir=tmp_path, warm_models=False)
    init_db(settings)
    with db(settings) as connection:
        connection.execute(
            "INSERT INTO eval_runs (id, created_at, status, request_json) "
            "VALUES ('stale', 'now', 'running', '{}')"
        )
    recover_stale_runs(settings)
    with db(settings) as connection:
        row = connection.execute("SELECT status, error FROM eval_runs WHERE id='stale'").fetchone()
    assert row["status"] == "failed"
    assert "interrupted" in row["error"]
```

- [ ] **Step 4: Run the complete fast suite and build**

```bash
cd backend && uv run pytest -v
cd frontend && npm test && npm run build
```
Expected: all fast tests pass; Vite build succeeds.

- [ ] **Step 5: End-to-end manual run**

1. `cd backend && uv sync --python 3.12`
2. `uv run python ../scripts/download_models.py --with-clip` (downloads ~1 GB; skip `--with-clip` to test V0–V2 only)
3. `cd backend && uv run python ../scripts/smoke_models.py --with-clip`
   - Confirm DINOv2 dim 384, CLIP dim 512, SAM2 3 candidates, hand asset loads, and timings print.
4. Start `scripts/run_backend.sh` and `scripts/run_frontend.sh`; open `http://127.0.0.1:5173`.
5. `curl -s 127.0.0.1:8000/api/health | python -m json.tool`
   - Confirm MPS/CPU device, assets, and model statuses.
6. Enroll item A: create it, capture 3+ gallery angles in session G1 (held in hand), then 5 test images in a different session T1 with a different angle/distance. Confirm every capture is accepted and thumbnails/masks appear.
7. Repeat for items B and C, ideally with visually similar packaging.
8. Scan A/B/C with V0/V1/V2 selected; confirm ~1–2 s per analysis, top-3 cards, prices, mask overlay, and stage timings.
9. Remove your hand from view → confirm `No hand detected`, not a stale result. Show a non-enrolled item → confirm the app returns matches/candidates without crashing.
10. Evaluate with V0/V1/V2 (and V3); confirm progress, metrics table, CI, `k_effective` note, per-item results, and failure links.
11. Restart the backend; confirm items, captures, and completed runs persist. Delete an item; confirm it disappears from Enroll/Scan while the completed run history remains.
12. For the real decision: repeat with 8–12 similar items, 3+ gallery and 10–20 test captures each, from separate sessions; record V2 vs V0/V1 accuracy, mask acceptance rate, and inspect the failure gallery.

- [ ] **Step 6: Commit**

```bash
git add backend/tests/test_models_slow.py backend/tests/test_hardening.py backend/app
git commit -m "test: slow model checks and hardening"
```

---

## Self-Review Notes

- **Spec coverage:** hand-aware localization (Tasks 4–5, 9, 11–13), V0–V3 variants (Task 13), lazy embeddings + exact cosine (Tasks 7, 15), enrollment capture rejection (Task 14), batch evaluation with metrics/CI/failure gallery (Tasks 8, 16), three-tab UI (Tasks 17–20), model download/smoke (Task 21), MPS fallback/health (Tasks 12, 22).
- **No live evaluation:** only the Scan tab streams; Evaluate is a batch background run.
- **SAM2 is not an embedder:** DINOv2/CLIP are the only retrieval vectors.
- **HNSW deferred:** `VectorIndex` protocol + `ExactCosineIndex` only.
- **Fast test isolation:** fake registries/pipelines mean `pytest` needs no model downloads; real models are gated behind `RUN_MODEL_TESTS=1`.
- **Version caution:** `transformers>=5.0,<6` and `mediapipe>=1.0,<2` are floors, not verified pins. `uv lock` records the exact resolved versions at Task 1; if SAM2/MediaPipe APIs differ, fix the wrappers at Tasks 9–11 and keep the interfaces unchanged.
- **DINOv2 licensing:** flagged in README; confirm before commercial use.


