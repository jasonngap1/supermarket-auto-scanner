# Supermarket Auto-Scanner

Local scanner app: hold an item in front of the camera, the app recognizes it,
and adds it to a running cart with quantity, prices, and an order total.

Recognition uses the whole camera frame with a DINOv2 embedding and exact
cosine matching. SAM2 and face detection are used only when enrolling an item,
to create a segmented gallery thumbnail for the cart.

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

The download is roughly 250 MB: SAM2, DINOv2, and the MediaPipe face detector.
Weights, model assets, and the SQLite database live under `data/` (gitignored).

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
   per item. Hold the item in the centre of the frame while capturing. The
   stored thumbnail is segmented and is used as the cart image.
2. Open **Scan** and show one item at a time to the camera. Scanning starts
   automatically.
3. The cart shows the item image, name, quantity, unit price, line total, and
   order total. Use the remove button on a line or **Clear** for the whole cart.

### Duplicate guard

A match is added at cosine similarity **0.80** or above. The scanner then locks
that item so holding it in view does not add it repeatedly.

The lock is released when the match drops below **0.60** (or no match is found)
for **3 consecutive scan frames**. It can also switch directly when a different
item scores at least 0.80 for 3 consecutive frames.

Consequence: to scan the next unit of the *same* item, remove the current item
from view for about 3 frames (~3.3 seconds), then show the next unit. Two
physically identical units swapped with no visible gap cannot be distinguished.
The thresholds live at the top of `frontend/src/tabs/scan_logic.js`.

## Storage

SQLite at `data/app.db`; media under `data/media`. Gallery captures, masks, and
segmented thumbnails are stored per item. Enrollment rejects duplicate images
and frames where the centre localizer finds no item mask.

Existing databases from the earlier validation lab keep their old tables and
V1/V2/V3 embeddings; the scanner ignores them.

## Tests

```bash
cd backend && uv run pytest -q
cd frontend && npm test && npm run build
```

Slow model checks, after downloading weights:

```bash
cd backend && RUN_MODEL_TESTS=1 uv run pytest -m slow -q
```

## Notes

- Whole-frame matching can use background context. Keep the scanner location
  and lighting reasonably stable for the best results.
- The frontend shows no model names, scores, or timings.
- DINOv2 weight licensing must be verified before commercial use.
- Local-only: backend binds to `127.0.0.1`; data never leaves the machine.
