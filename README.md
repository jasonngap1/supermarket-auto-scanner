# Supermarket Auto-Scanner — Model Validation Lab

Local prototype for recognizing a supermarket item held in front of a laptop
camera and retrieving its name and price. It exists to answer whether
YOLO-prompted segmentation and DINOv2 retrieval are accurate enough for a final
scanner.

## Stack

- FastAPI + SQLite + local files; all inference runs server-side.
- YOLO26n (`yolo26n.pt`) detects the item bounding box. COCO class labels are
  only used to filter detections, never for recognition.
- SAM2 (`facebook/sam2.1-hiera-tiny`) segments the YOLO box using the box plus
  its centre point. The best candidate is selected by predicted IoU, overlap
  with the YOLO box, area agreement, and border leakage.
- Embeddings: DINOv2 (`facebook/dinov2-small`), 384 dimensions.
- Matching: exact cosine, top-3 live, top-5 in evaluation.
- Vite vanilla JavaScript UI: **Scan**, **Enroll**, **Evaluate**.
- Python 3.12 via `uv`; Apple M1 with MPS, CPU fallback.

Variants:
- `v0` — whole-frame DINOv2 control. Detects background shortcutting.
- `v2` — YOLO box -> SAM2 mask -> DINOv2. The production path.

## Setup

```bash
cd backend
uv sync --python 3.12
uv run python ../scripts/download_models.py
cd ../frontend
npm install
```

The download fetches SAM2, DINOv2, and `yolo26n.pt` (about 250 MB total) and
writes `data/model_manifest.json`. Weights, media, and SQLite live under
`data/` (gitignored).

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

## How localization works

1. YOLO detects objects in the frame.
2. One detection is selected: confidence first, minus a centre-distance
   penalty. Detections are rejected when they are too small or too large,
   touch the frame border, or belong to `AUTO_SCANNER_YOLO_REJECT_CLASSES`
   (default `person`).
3. SAM2 is prompted with the selected box and its centre point and returns
   three mask candidates.
4. A mask is rejected when its predicted IoU is below
   `AUTO_SCANNER_YOLO_SAM_MIN_PRED_IOU` (0.50), its area is outside the YOLO
   area limits, it touches the border too much
   (`AUTO_SCANNER_YOLO_SAM_MAX_BORDER_FRACTION`, 0.10), its bbox IoU with the
   YOLO box is below `AUTO_SCANNER_YOLO_SAM_MIN_BOX_IOU` (0.40), or less than
   `AUTO_SCANNER_YOLO_SAM_MIN_BOX_COVERAGE` (0.75) of its pixels are inside the
   YOLO box.
5. The mask crop is letterboxed to 224x224 and embedded with DINOv2.

No object name from YOLO is used to identify the item. Only the box matters.

## Enrollment

Enrollment stores the image and a plain thumbnail. It does not run YOLO or
SAM2, and it does not require a mask. The `v2` vector is computed lazily when
the gallery or a test capture is first embedded. A capture where YOLO or SAM2
fails simply has no `v2` vector and reports `no_detection` or
`mask_rejected` in Scan and Evaluate.

Near-duplicate warnings compare whole-frame `v0` vectors, so enrollment stays
fast and independent of detection.

Existing databases keep working. New captures write the legacy columns as
`mask_rel_path=""`, `mask_found=0`, `mask_status="not_localized"`, and
`localization_json="{}"`. No schema migration is required.

## Performance on the M1

Measured on the current stored captures with `scripts/benchmark_yolo.py`:

- YOLO detection: p50 ~24 ms, p95 ~27 ms.
- SAM2 segmentation: ~0.44 s per accepted frame.
- Warm DINOv2: ~20 ms.
- End-to-end `v2` on an accepted frame: roughly 0.45–0.5 s.
- Live analysis refreshes on request; the frontend auto-scan runs about once
  per second.

Detection coverage is the limiting factor. On the current five stored
captures, YOLO detected two and produced a valid SAM2 mask for both. A product
outside the 80 COCO classes produces no binding. Use
`scripts/benchmark_yolo.py --conf` and `--reject-classes` to compare settings.

## Evaluation protocol

1. **Enroll** each item with 3+ gallery captures from multiple angles in
   session `G1`.
2. Capture 5–10 **test** images in a different session `T1` with different
   angles, lighting, and distance. Never put test images in the gallery.
3. Run **Evaluate** with `v0` and `v2`.
4. Read top-1/top-5, the 95% Wilson interval, per-item results, confusion,
   mask-accepted rate, and the failure gallery. If `v0` is much better than
   `v2`, suspect background shortcutting or mask failures.

Minimum for a weak signal: 5 items × 3 gallery × 5 test. For a credible
go/no-go: 8–12 visually similar items × 10–20 test captures each.

Evaluation uses the same stored images. `v2` localization is recomputed on
first use and cached with a version that includes the YOLO and SAM2 settings,
so changing a threshold invalidates only `v2` vectors.

Near-duplicate test images still inflate scores; every report counts them.

## Tests

```bash
cd backend && uv run pytest -v
cd frontend && npm test && npm run build
cd backend && RUN_MODEL_TESTS=1 uv run pytest -m slow -v
# Include the YOLO weight test:
cd backend && RUN_MODEL_TESTS=1 RUN_YOLO_TESTS=1 uv run pytest -m slow -v
```

## Notes

- Evaluation is batch/offline; Scan is live inference and reports no accuracy.
- Repeated evaluation runs read cached vectors, so localization latency is
  only accurate on the first run after a cache version change.
- Failure artifacts are the raw frame plus truth/prediction text; they do not
  re-render the live mask.
- Matching is exact cosine. HNSW is deferred behind `VectorIndex`.
- Ultralytics and the YOLO26 weights are AGPL-3.0. This is acceptable for the
  local lab; commercial distribution requires a licence review or a commercial
  licence.
- DINOv2 weight licensing must be verified before any commercial use.
- Legacy columns for hand and mask metadata remain in existing databases but
  are no longer written by enrollment.
