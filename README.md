# Supermarket Auto-Scanner

Local scanner app: hold an item in front of the camera, the app recognizes it,
and adds it to a running cart with quantity, prices, and an order total.

Recognition uses YOLO26n to find the item box, SAM2 to segment that box, and
DINOv2 to embed the mask crop. The same chain runs when enrolling a gallery
photo, so gallery and live vectors describe the same view.

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

The download is roughly 250 MB: YOLO26n, SAM2, and DINOv2. Weights, caches, and
the SQLite database live under `data/` (gitignored).

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
   per item. Show one item at a time. A frame is stored only when YOLO finds an
   item box and SAM2 returns an accepted mask; the segmented thumbnail is used
   in the cart.
2. Open **Scan** and show one item at a time to the camera. Scanning starts
   automatically. A frame with no detection reports `no_detection`; it does not
   match against the gallery.
3. The cart shows the item image, name, quantity, unit price, line total, and
   order total. Use the remove button on a line or **Clear** for the whole cart.

### Detection pipeline

1. YOLO detects objects. One box is selected by confidence minus a
   centre-distance penalty; `person` is rejected by default, as are boxes that
   are too small, too large, or touch the frame border.
2. SAM2 is prompted with the box and its centre point and returns three mask
   candidates.
3. A mask is rejected when its predicted IoU, overlap with the YOLO box, area,
   or border leakage is outside the configured limits.
4. The mask crop is letterboxed to 224x224 and embedded with DINOv2.

YOLO class names are only used for filtering, never for recognition. Class
coverage is the limiting factor: a product outside the 80 COCO classes produces
no detection. Tune `AUTO_SCANNER_YOLO_CONF` and the mask thresholds in
`.env.example` if needed.

### Enrollment

Enrollment runs YOLO -> SAM2 and stores the image, mask, and segmented
thumbnail only when the frame has both a bounding box and an accepted mask.
Frames with no detection are rejected with `no_detection`; frames where the
mask fails selection are rejected with `mask_rejected`. Retake with one item
centred in the frame.

The gallery vector is computed on first scan from the stored mask.

### Duplicate guard

A match is added at cosine similarity **0.60** or above. The scanner then locks
that item so holding it in view does not add it repeatedly.

The lock is released when the match drops below **0.60** (or no match is found)
for **3 consecutive scan frames**. It can also switch directly when a different
item scores at least 0.60 for 3 consecutive frames.

Consequence: to scan the next unit of the *same* item, remove the current item
from view for about 3 frames (~3.3 seconds), then show the next unit. Two
physically identical units swapped with no visible gap cannot be distinguished.
The thresholds live at the top of `frontend/src/tabs/scan_logic.js`.

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

Measured with the validation lab's benchmark on five stored captures:

- YOLO detection: p50 ~25 ms.
- SAM2 segmentation: ~0.42 s per accepted frame.
- Warm DINOv2: ~20 ms.
- End-to-end mask vector: roughly 0.45-0.5 s.

The frontend scans about once per second and skips overlapping requests.

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

- `v2` (YOLO box -> SAM2 mask -> DINOv2) is the only matching view. The
  whole-frame control variant remains in the pipeline code for comparison.
- The frontend shows no model names, scores, or timings.
- Matching is exact cosine. HNSW is deferred behind `VectorIndex`.
- Ultralytics and the YOLO26 weights are AGPL-3.0. This is acceptable for the
  local app; commercial distribution requires a licence review or a commercial
  licence.
- DINOv2 weight licensing must be verified before commercial use.
- Local-only: backend binds to `127.0.0.1`; data never leaves the machine.
