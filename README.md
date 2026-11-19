# Supermarket Auto-Scanner

Local scanner app: hold an item in front of the camera, the app recognizes it,
and adds it to a running cart with quantity, prices, and an order total.

Recognition uses YOLO26n-seg to find the item box and its instance mask in one
ONNX Runtime pass, then DINOv2 to embed the normalized mask crop. The same chain
runs when enrolling a gallery photo, so gallery and live vectors describe the
same view.

## Requirements

- Python 3.12 via `uv`
- Node 24
- Apple silicon is supported through MPS, with CPU fallback

## Setup

```bash
cd backend
uv sync --python 3.12
uv run python ../scripts/download_models.py

cd ../frontend
npm install
```

The download fetches DINOv2 and `yolo26n-seg.pt`, then exports it to
`data/weights/yolo26n-seg.onnx` with `imgsz=640` (about 250 MB total). Weights,
caches, and the SQLite database live under `data/` (gitignored). Re-run the
script after changing the YOLO model or export settings.

## Run

Terminal 1:

```bash
scripts/run_backend.sh
```

Terminal 2:

```bash
scripts/run_frontend.sh
```

Open http://127.0.0.1:5173. `GET http://127.0.0.1:8000/api/health` reports model
and device status.

## Use

1. Open **Items**, add a name and price, and capture at least one gallery photo
   per item. Show one item at a time. A frame is stored only when YOLO returns
   both an item box and an accepted mask; the segmented thumbnail is used in
   the cart. Item names and prices are editable after enrollment.
2. Open **Scan** and show one item at a time to the camera. Scanning starts
   automatically. A frame with no detection reports `no_detection`; it does not
   match against the gallery.
3. The cart shows the item image, name, quantity, unit price, line total, and
   order total. Use the remove button on a line or **Clear** for the whole cart.

### Detection pipeline

1. YOLO26n-seg returns boxes plus masks at the original frame size. One box is
   selected by confidence minus a centre-distance penalty; `person` is rejected
   by default, as are boxes that are too small, too large, or touch the frame
   border.
2. The selected detection's mask is rejected when it is empty, its area or
   border leakage is outside the configured limits, or its agreement with the
   detection box is too loose.
3. The mask crop is normalized: masked-pixel channel means are equalized
   (gray-world white balance), then each channel is standardized to mean 128
   and standard deviation 50. The same step runs for gallery and live frames.
4. The normalized crop is letterboxed to 224x224 and embedded with DINOv2.

YOLO class names are only used for filtering, never for recognition. Class
coverage is the limiting factor: a product outside the 80 COCO classes produces
no detection. Tune `AUTO_SCANNER_YOLO_CONF` and the mask thresholds in
`.env.example` if needed.

### Enrollment

Enrollment runs YOLO26n-seg and stores the image, mask, and segmented thumbnail
only when the frame has both a bounding box and an accepted mask. Frames with
no detection are rejected with `no_detection`; frames where the mask fails
selection are rejected with `mask_rejected`; model failures report
`model_error`. Retake with one item centred in the frame.

The gallery vector is computed on first scan from the stored mask.

### Duplicate guard

A match is added at cosine similarity **0.70** or above only after it holds for
**0.5 s**, so a frame with both a bbox and a mask has shown a settled item before
it can register. A detector dropout during that window over an unchanged view
does not restart the wait, and an add always uses a positive (bbox + mask) frame.
The scanner then locks that item so holding it in view does not add it repeatedly.

A missed frame does not by itself release the lock. Release needs the item to be
gone for **1.2 s** *and* the camera view to have changed from the locked
presentation, compared through an 8x8 luma signature that ignores global
brightness shifts. A frame where YOLO finds an object but rejects its mask also
counts as still in view. A detector dropout over an unchanged view therefore
keeps the lock and cannot add the same unit twice. A different item scoring at
least 0.70 for 3 consecutive frames switches the lock directly.

Consequence: to scan the next unit of the *same* item, move it out of view for
about 1.2 s, then show the next unit. Two physically identical units swapped
with no visible gap cannot be distinguished. The thresholds live at the top of
`frontend/src/tabs/scan_logic.js`.

## Storage

SQLite at `data/app.db`; media under `data/media`. Gallery images, masks, and
thumbnails are stored per item. Enrollment rejects duplicate images and frames
without an accepted item mask.

Existing databases keep working. Older captures may still have an empty mask
path. No schema migration is required.

### Shared enrolled catalog

`catalog/` is committed to the repo. It holds `catalog.json` plus every gallery
image, mask, and thumbnail needed to recreate the enrolled item list. A fresh
clone with no `data/app.db` imports this catalog on first startup, so another
user sees the same Items list and can scan the same products.

The running app refreshes `catalog/` after every item or gallery-capture change.
Commit those changes to share newly enrolled items. To rebuild the snapshot
manually:

```bash
cd backend
uv run python ../scripts/export_catalog.py
```

`catalog/` takes priority only when the local database does not exist. An
existing local database is never overwritten; delete it only if you intend to
replace local items with the repository snapshot.

Older catalog entries without a mask are still seeded without one. Their
gallery vectors are recomputed with the current detection settings on first
scan and skipped when no mask can be found.

## Performance on the M1

Measured on the current stored captures with `scripts/benchmark_yolo.py`:

- YOLO26n-seg ONNX: p50 ~77 ms, p95 ~107 ms per frame (CPU ExecutionProvider).
- Warm DINOv2: ~20 ms.
- End-to-end `v2` on an accepted frame: roughly 100 ms.

The frontend scans every 250 ms and skips overlapping requests. ONNX Runtime
runs on CPU; the `.pt` MPS detector was faster per call but the segmenting
model replaces a separate SAM2 stage, so the full loop is much shorter.

### Compute footprint

Measured on an 8-core Apple M1 with 16 GB RAM, `device=auto` (DINOv2 on MPS,
YOLO ONNX on CPU), with the current gallery of 8 items and 22 captures:

- Backend, models warm and idle: about **435 MB** physical footprint.
- Backend after scanning: about **585 MB** peak.
- Vite dev server: about **50 MB**.
- Idle CPU: near zero.
- One accepted scan frame: about **0.11-0.15 s** wall-clock. It costs roughly
  **0.6 CPU-seconds**, spread across the ONNX Runtime thread pool, so a scan
  burst can occupy about 4 cores briefly. Scanning every 250 ms averages around
  2.4 cores while the **Scan** tab is open. Raise `SCAN_INTERVAL_MS` in
  `frontend/src/tabs/scan.js` to lower the average proportionally.

No GPU is required; `AUTO_SCANNER_DEVICE=cpu` forces CPU-only for both models.
Under 0.7 GB of server memory is enough for the current catalog. Keep the `v3`
SAM2 variant unselected on memory-limited hosts: it loads a second, larger model
on first use. The browser tab and camera pipeline are outside these figures.

## Tests

```bash
cd backend && uv run pytest -q
cd frontend && npm test && npm run build
```

Slow model checks, after downloading weights:

```bash
cd backend && RUN_MODEL_TESTS=1 uv run pytest -m slow -q
# Include the YOLO weight check:
cd backend && RUN_MODEL_TESTS=1 RUN_YOLO_TESTS=1 uv run pytest -m slow -q
```

## Notes

- `v2` (YOLO26n-seg box + mask -> normalized crop -> DINOv2) is the only
  matching view. The whole-frame control variant remains in the pipeline code
  for comparison.
- The ONNX export pins `imgsz=640`. Changing `AUTO_SCANNER_YOLO_IMGSZ` requires
  re-running `scripts/download_models.py` and invalidates `v2` vectors.
- Photometric normalization and the YOLO-seg switch changed the embedding
  distribution, so the 0.70 add threshold may need retuning.
- The frontend shows no model names, scores, or timings.
- Matching is exact cosine. HNSW is deferred behind `VectorIndex`.
- Ultralytics and the YOLO26 weights are AGPL-3.0. This is acceptable for the
  local app; commercial distribution requires a licence review or a commercial
  licence.
- DINOv2 weight licensing must be verified before commercial use.
- Local-only: backend binds to `127.0.0.1`; data never leaves the machine.
