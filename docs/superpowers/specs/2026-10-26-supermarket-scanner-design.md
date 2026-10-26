# Supermarket Auto-Scanner — Model Validation Prototype

## Context

Goal: find out whether **hand-aware SAM2 segmentation + DINOv2 embeddings** can reliably recognize a supermarket item held up to a laptop camera and retrieve its name and price. The repo is empty. This prototype is a *model validation lab* and simultaneously the reusable skeleton for the final scanner app.

Environment: Apple M1, 16 GB, Node 24, `uv` available. System Python is 3.14 — too new for MediaPipe, so pin **Python 3.12 via uv**.

Decisions already made with the user:
- Automatic held-item localization via **MediaPipe HandLandmarker → SAM2 prompts** (no click by default).
- **Enrollment via webcam** (name + price, multi-angle gallery captures, plus separate held-out test captures).
- **Built-in evaluation** with top-1/top-5 accuracy and per-stage latency across pipeline variants.
- Accuracy first; ~1 fps live analysis (~1–2 s/frame) is acceptable.
- Local monolith: FastAPI backend, Vite + vanilla JS frontend, SQLite + local files.
- Exact cosine matching now behind a `VectorIndex` interface; **HNSW deferred** (not needed at prototype scale).

## MVP behavior

Three tabs:

1. **Scan** — live webcam preview; auto-analyze toggle at ~1 fps; annotated snapshot; top-3 item cards (name, price, score); per-stage timing table; explicit `no_hand` / `mask_rejected` / `no_gallery` states.
2. **Enroll** — create item (name, price); capture gallery and test images via webcam; show mask-overlay thumbnails and hand/mask status; delete captures/items; gallery/test counts.
3. **Evaluate** — one-click background run; metrics table comparing variants; per-item accuracy; confusion; failure gallery with mask overlay and rejection reason.

**Evaluation is batch, not real-time.** The Scan tab is the live ~1 fps path and reports only matches, similarity, and latency — live frames have no ground-truth label, so scanning shows no accuracy. Evaluate runs on demand over labeled, held-out captures in a background job, keeping results repeatable and preventing tuning against the test set. Scanning pauses during an evaluation run (shared inference gate).

Success criteria:
- End-to-end works on the M1: enroll 5+ items, scan live, run evaluation, persist across restart.
- The evaluation answers the go/no-go question: does V2 (SAM2 + DINOv2) beat V0 (whole frame) and V1 (hand crop), and does DINOv2 beat CLIP? Also reports how often segmentation is usable (mask acceptance rate).

## Architecture

```
Browser (Vite vanilla JS)
   │  HTTP: 1 frame/sec JPEG + JSON, polling for eval
   ▼
FastAPI (one uvicorn worker, 127.0.0.1)
   │  one global inference gate (threading.Lock)
   ├─ MediaPipe HandLandmarker ──► prompt geometry ──► SAM2 (sam2.1-hiera-tiny)
   │                                                   └─ mask selection
   ├─ preprocess crop ──► DINOv2-small / CLIP-B32 ──► embeddings
   ├─ ExactCosineIndex ──► item-level top-k
   └─ SQLite + data/ (images, masks, thumbs, eval artifacts)
```

- `PYTORCH_ENABLE_MPS_FALLBACK=1` must be set **before `import torch`**: set in the run script and defensively at the top of `backend/app/bootstrap.py`, imported first by `main.py`.
- Device auto-select: MPS if available, per-model fallback to CPU on error. Float32 only, no autocast/bf16 on MPS.
- One uvicorn worker; all model calls serialized through one inference gate. Evaluation acquires the gate per sample.
- Lazy-load SAM2 and CLIP on first use; warm hand + DINOv2 at startup. Health endpoint exposes configured/actual device, loaded state, and last error.
- `torch.inference_mode()` everywhere; discard the first-sample latency from metrics.

## Held-item pipeline

1. Decode frame → RGB. MediaPipe HandLandmarker (Tasks API, `RunningMode.IMAGE`, `num_hands=1` — single held item assumption).
2. Landmark geometry (normalized → pixels):
   - `hand_scale` = max(hand bbox width, hand bbox height)
   - `palm_center` = mean of landmarks 0, 5, 9, 13, 17
   - `pinch_center` = midpoint(thumb tip 4, index tip 8)
   - `grasp_center` = midpoint(thumb tip 4, midpoint(index tip 8, middle tip 12))
   - positive point = `pinch_center`; when thumb–index distance < `0.35 × hand_scale`, use `grasp_center`
   - negative points = wrist 0, `palm_center`, MCP joints 5, 9, 13, 17 (**never fingertips** — they touch the item)
   - box = bbox(thumb 4, index 8, middle 12, palm_center) expanded by `0.35 × hand_scale`; prompt-box mode is configurable (`pinch_roi` default; A/B against `none` and expanded hand box)
3. SAM2 `multimask_output=True` → 3 candidates. Select with **hard filters, then rank**:
   - reject if positive point not inside mask
   - reject if wrist or palm center inside mask
   - reject if > 6 of 21 landmarks inside mask (start value, configurable)
   - reject if mask area < 0.3% or > 30% of frame
   - reject if mask bbox does not overlap the grasp ROI
   - rank survivors: `pred_iou − 0.05 × landmark_hits − area_penalty`, where `area_penalty = 0.05 × max(0, area_ratio − 0.15) / 0.15`; tie-break toward smaller mask
   - if none survive: fall back to the V1 hand-box crop and report `mask_rejected`
   - store all 3 candidates + scores in `localization_json` for tuning/debug
4. Localization happens **once per frame** and is shared by all variants.
5. Preprocessing: mask bbox + 10% margin; gray out pixels outside the mask; letterbox to 224×224 (same for gallery and scan). Store `model_key` and `preprocess_version` with every embedding so caches are never silently mixed. Optional cheap experiment: `v2_crop_only` (no gray fill).
6. Embeddings: DINOv2 CLS token (`facebook/dinov2-small`, 384-d), CLIP image embedding (`openai/clip-vit-base-patch32`, 512-d), L2-normalized.
7. Match: exact cosine; item score = max over that item's gallery captures; return top-k items. Empty gallery → `no_gallery`.

### Variants

| ID | Preprocessing | Embedding |
|----|---------------|-----------|
| V0 | whole frame | DINOv2 |
| V1 | hand-box crop, no SAM2 | DINOv2 |
| V2 | SAM2 mask crop + gray fill | DINOv2 |
| V3 | SAM2 mask crop + gray fill | CLIP (lazy, optional) |

**SAM2 is a segmenter, not an embedder.** It outputs a mask; it has no CLIP head, and its Hiera features are not used as retrieval vectors. Retrieval vectors come only from DINOv2 or CLIP applied to the segmented crop. DINOv2 is the primary embedder (self-supervised visual similarity, finer 14 px patches suited to SKU-level detail); CLIP ViT-B/32 is a semantic baseline (image–text training, coarser 32 px patches). No fused DINOv2+CLIP variant in v1.

Annotated response JPEG: hand bbox, selected mask overlay, best item name/price, per-variant timings.

## Data model

SQLite (WAL, foreign keys) + files under `data/`:

- `items(id, name, price_cents CHECK >= 0, created_at)` — integer cents; currency symbol from config.
- `captures(id, item_id FK cascade, role gallery|test, session_id, rel_path, thumb_rel_path, mask_rel_path, sha256, width, height, hand_found, mask_found, mask_status, localization_json, created_at)`; unique `(item_id, role, sha256)`; reject duplicate images.
- `embeddings(capture_id FK cascade, variant, model_key, preprocess_version, dim, vector BLOB float32, created_at)`, PK on the first five columns.
- `eval_runs(id, status queued|running|completed|failed, request_json, progress_done/total, metrics_json, timestamps, error)`.
- `eval_samples(run_id FK, variant, true/pred item + name + price snapshot, topk_json, correct_top1/top5, sample_status, timings_json, rendered_rel_path)`.

Files: `data/images/{item_id}/{uuid}.jpg`, `masks/{capture_id}.png`, `thumbs/`, `runs/{run_id}/`, `hf_cache/`. All gitignored. Generated UUID filenames; cap upload size; serve only from the data root.

Matching is behind `VectorIndex` (`add`, `search`, `remove`, rebuildable from DB). `ExactCosineIndex` uses NumPy. HNSW can be added later without changing callers.

## REST API

All JSON; Vite proxies `/api` and `/media` to `127.0.0.1:8000` (no CORS needed). Bind backend to localhost only.

```
GET    /api/health                       models, devices, asset + version status
GET    /api/meta                         variants, currency, thresholds, model keys
GET    /api/items                        list with gallery/test counts
POST   /api/items                        {name, price_cents}
DELETE /api/items/{id}
GET    /api/items/{id}/captures?role=
POST   /api/items/{id}/captures?role=gallery|test&session_id=...   multipart files[]
DELETE /api/items/{id}/captures/{cid}
POST   /api/analyze                      multipart image + variants
POST   /api/evaluations                  {variants, item_ids?} → 202 {run_id}
GET    /api/evaluations                  list runs
GET    /api/evaluations/{run_id}         progress, metrics, per-item, confusion
GET    /api/evaluations/{run_id}/failures?variant=&limit=
GET    /media/...                        images/thumbs/masks/artifacts
```

`/api/analyze` per-variant statuses: `ok`, `no_hand`, `mask_rejected`, `no_gallery`, `model_error`. Invalid upload → 422; model unavailable after fallback → 503. Scan failures still return 200 with a per-variant status.

Evaluation is a background thread/job; `GET` polls progress. On startup, stale `running` runs are marked interrupted.

## Frontend

Vite vanilla JS, one shared camera module (`getUserMedia` 1280×720, canvas → JPEG quality 0.9) used by both Enroll and Scan. One in-flight analysis at a time — skip frames while busy. No WebSockets, charts, auth, or history browser.

Files: `frontend/src/{main.js, api.js, camera.js, tabs/{scan,enroll,evaluate}.js, style.css}` + Vite proxy config.

Enrollment rules:
- Gallery and test captures must both be held-in-hand, using the same capture code/resolution/quality. Separate `session_id` per batch; test captures should use different angles/lighting/distance.
- Reject captures with no hand or rejected mask (ask for retake) so every stored image has a valid localization artifact.
- Show mask thumbnail + status; warn on near-duplicate test/gallery embeddings (> 0.995 cosine).
- Tune mask thresholds only on separate debug frames, never on the evaluation test set.

## Evaluation methodology

Evaluation is **offline/batch over labeled test captures**, never streaming or continuously updated during scanning. The live Scan path reports confidence and timing only.

- Build gallery embeddings per variant from gallery captures only; query with test captures.
- Report: micro + macro top-1, top-5 (with `k_effective = min(5, n_items)` warning when fewer than 5 items), 95% Wilson CI, per-item accuracy, confusion, similarity margin (correct score − best incorrect), hand-found %, mask-accepted %, failure reasons, latency p50/p95 total and per stage.
- Failure gallery: original frame, mask overlay, true item, predicted top-k, rejection reason; artifacts copied into the run directory so item deletion doesn't break history.
- Minimum data: 5 items × 3+ gallery × 5+ test for a weak signal; 8–12 visually similar items × 10–20 test captures for a credible go/no-go. Show an "insufficient data" banner below these thresholds.
- Keep V0/V1 as controls: if V0 ≫ V2, suspect background shortcut and inspect failures.

## File structure

```
backend/
  .python-version                     # 3.12
  pyproject.toml / uv.lock
  app/
    bootstrap.py                      # env vars before torch
    main.py                           # create_app(), lifespan, routers
    config.py  db.py  schema.sql  domain.py  schemas.py  dependencies.py
    api/        health.py items.py captures.py analyze.py evaluations.py media.py
    services/   storage.py items.py captures.py analysis.py evaluation.py
    pipeline/   types.py hand.py prompts.py sam2.py mask_select.py
                preprocess.py embedders.py variants.py annotate.py frame_pipeline.py
    ml/         devices.py registry.py
    matching/   vector_index.py matcher.py
    evaluation/ metrics.py runner.py artifacts.py
  assets/hand_landmarker.task         # downloaded, gitignored
  tests/                              # unit + fake-pipeline API + slow model test
frontend/                             # Vite vanilla JS
scripts/  download_models.py run_backend.sh run_frontend.sh smoke_models.py
data/                                 # gitignored
```

Keep CV wrappers free of FastAPI imports and API modules free of torch/mediapipe imports, so fake-pipeline tests run without weights.

## Implementation phases

**M0 — Toolchain & skeleton.** Pin Python 3.12 with uv; backend/app factory + bootstrap + health; Vite app with proxy; webcam preview. Verify imports, MPS availability, env-var-before-torch, both servers start.

**M1 — Storage & enrollment plumbing (no models).** Schema, item CRUD, UUID file storage, thumbnails, capture roles/sessions, cascade delete, media serving. Verify repository/API tests, persistence across restart.

**M2 — V0 end-to-end.** DINOv2 wrapper, preprocessing, exact cosine index, item matcher, Scan tab. Verify embedding shape 384 / norm ≈ 1, top-3 on a seeded gallery, first-load excluded.

**M3 — Hand localization + V1.** MediaPipe wrapper, prompt geometry, hand crop, debug overlay, `no_hand` state. Verify prompt-geometry unit tests and hand-found rate on live frames.

**M4 — SAM2 + V2.** SAM2 wrapper, 3-candidate selection, fallback to V1, mask thumbnails, localization JSON. Verify on 20–30 held-item frames; compare prompt modes; determinism across repeated runs; MPS vs CPU on a few frames.

**M5 — CLIP + model lifecycle.** Lazy loading, device fallback, health/meta, `model_error` surfaced cleanly. Verify dimensions 384/512, selective loading, health states.

**M6 — Evaluation runner.** Background job, progress polling, cached localization/embeddings, metrics/CI/confusion, failure artifacts, evaluation UI.

**M7 — Enrollment UX & auto-scan.** Separate gallery/test sections, session IDs, capture rejection, mask thumbnails, counts, one-in-flight auto-analysis, error states.

**M8 — Hardening & end-to-end.** Upload limits, path safety, logging, stale-run recovery, README, download/smoke scripts, final manual run.

After approval, also copy this spec to `docs/superpowers/specs/2026-10-26-supermarket-scanner-design.md` and commit it before implementation.

## Dependencies & assets

- `uv` + `.python-version` 3.12; commit `uv.lock`. Backend: fastapi, uvicorn, python-multipart, pydantic-settings, pillow, numpy, torch, torchvision, transformers, mediapipe, huggingface-hub. Dev: pytest, pytest-timeout, httpx.
- MediaPipe **1.x** (macOS arm64 wheel, supports 3.12); use Tasks API, `RunningMode.IMAGE`. Download the immutable hand model asset:
  `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`
  Verify size/SHA-256 at startup. MediaPipe pulls `opencv-contrib-python`; do **not** additionally install `opencv-python`.
- HF models: `facebook/sam2.1-hiera-tiny`, `facebook/dinov2-small`, `openai/clip-vit-base-patch32`. Pin revisions and download via `huggingface_hub.snapshot_download` into `data/hf_cache`; record a manifest. CLIP (~600 MB) is optional until V3 is enabled.
- Transformers 5.x SAM2 API must be verified at M0/M4 — do not follow stale 4.x snippets. Use `Sam2Processor` + `Sam2Model`, post-process masks to original size.
- Env: `PYTORCH_ENABLE_MPS_FALLBACK=1`, `HF_HOME=data/hf_cache`, `TOKENIZERS_PARALLELISM=false`, single worker.
- Frontend: Vite vanilla template, Node 24; commit `package-lock.json`.
- Licensing note: SAM2 Apache-2.0, CLIP MIT, MediaPipe Apache-2.0; **verify DINOv2 weight terms (historically non-commercial) before production**.

## Error handling, privacy, limits

- Clear per-variant statuses instead of silent bad matches; `no_hand` never returns a stale result.
- One inference gate prevents MPS contention between scan and evaluation; Scan shows "evaluation busy".
- Evaluation runs in the background with progress; per-sample gate acquisition.
- First-run model downloads/loading reported in health with a loading state.
- Local-only: bind 127.0.0.1, UUID filenames, upload size cap, data stays on disk.
- Model load failure → per-model CPU retry once, then `model_error` (no infinite retries).
- Price stored as integer cents; no float money.

## Testing & verification

Fast tests (no weights):
- `test_prompts.py` — synthetic 21 landmarks → points/box, fingertips not negatives, clipping, multi-hand.
- `test_mask_select.py` — rejection rules, ranking, fallback signal with synthetic masks.
- `test_preprocess.py` — margin crop, gray fill, letterbox, edge/empty masks, 224 shape.
- `test_vector_index.py` — normalization, cosine ranking, item-level max aggregation, `k > n_items`, empty gallery.
- `test_repository.py` — roles, cascade delete, BLOB round-trip, stale-run recovery.
- Fake-pipeline API tests — item/capture/analyze/evaluate flows without importing torch, transformers, or mediapipe.

Slow integration (`RUN_MODEL_TESTS=1`, `pytest.mark.slow`): hand detected; SAM2 masks non-empty and deterministic (MPS vs CPU loose IoU ≥ 0.9 or clean fallback); DINOv2/CLIP dimensions 384/512 normalized.

Manual E2E:
1. `cd backend && uv sync --python 3.12 && uv run python ../scripts/download_models.py`
2. Start backend (`scripts/run_backend.sh`, or `uv run uvicorn app.main:app --reload --port 8000`) and frontend (`npm ci && npm run dev`); `curl 127.0.0.1:8000/api/health` shows MPS/CPU, assets, loaded models; open `http://127.0.0.1:5173`.
3. Enroll 2–3 items: 3+ gallery captures across angles (session G1), 5 test captures in a different session (T1); confirm masks/thumbnails.
4. Scan each item with V0/V1/V2 enabled; confirm ~1–2 s latency, top-3 cards, annotated snapshot; remove hand → `no_hand`.
5. Run evaluation; confirm metrics table, CI, per-item rows, failure gallery, `k_effective` warning.
6. Restart backend; data and completed runs persist. Delete an item; history remains.
7. For the real decision, repeat with 8–12 similar items × 10–20 test captures each and record V2 vs V0/V1 and DINOv2 vs CLIP.

## Non-goals

Authentication, multi-user, cloud/remote access; HNSW/ANN; multiple items per frame or multiple hands; video tracking/SAM2 video mode; 30 fps/WebRTC; training/fine-tuning; threshold auto-tuning; live labeled evaluation sessions or continuously updated accuracy; fused DINOv2+CLIP embeddings; SAM2 encoder features as retrieval vectors; click-to-prompt in default flow; barcode/OCR; catalog/checkout integration; item editing/price history; export/charts; masked patch-token pooling (possible follow-up experiment); Docker/CI GPU setup; responsive/cross-browser polish.

## Key risks

| Risk | Mitigation |
|------|-----------|
| SAM2 segments the hand or hand+item | Negative points on palm/wrist/MCP; small grasp-ROI box; hard filters + mask acceptance rate; V1 fallback; inspect masks during enrollment |
| Gallery/test protocol mismatch creates fake accuracy | Held-in-hand for both; separate sessions/angles; reject no-hand captures; near-duplicate warning |
| V0 wins by matching background/desk | Report V0/V1 controls and failure gallery; don't tune on test set |
| CLIP loses partly because gray fill is out-of-distribution and ViT-B/32 loses fine print | Keep identical preprocessing for the fair comparison; if CLIP loses, retest V3 with original background and/or CLIP ViT-B/16 before concluding CLIP is unsuitable. Higher-resolution DINOv2 (518) is the follow-up if SKU detail is the bottleneck |
| SAM2 on MPS crashes/empty masks | fp32, inference gate, per-model CPU fallback, determinism test, health error surface |
| Big first-run downloads / model load latency | Pinned asset script, lazy CLIP/SAM2, loading states, exclude first sample |
| Python 3.14 incompatibility | uv-pinned 3.12; verify mediapipe wheel at M0 |
| DINOv2 license blocks commercialization | Flag now; validate before production |
| Too little data for credible metrics | Minimum-data banner, Wilson CI, realistic test-capture counts |
