# Supermarket Auto-Scanner — Model Validation Lab

Local prototype for recognizing a supermarket item held in front of a laptop
camera and retrieving its name and price. It exists to answer whether
YOLO-prompted segmentation and DINOv2 retrieval are accurate enough for a final
scanner.

## Stack

- FastAPI + SQLite + local files; all inference runs server-side.
- YOLO26n-seg (`yolo26n-seg.onnx`) returns the item bounding box and its
  instance mask in one ONNX Runtime forward pass. COCO class labels are only
  used to filter detections, never for recognition.
- SAM2 (`facebook/sam2.1-hiera-tiny`) remains available only for the `v3`
  comparison variant: YOLO box plus centre point prompts SAM2.
- Mask crops are normalized (gray-world white balance plus per-channel
  standardization) before embedding, so gallery and live frames land in the
  same photometric space.
- Embeddings: DINOv2 (`facebook/dinov2-small`), 384 dimensions.
- Matching: exact cosine, top-3 live, top-5 in evaluation.
- Vite vanilla JavaScript UI: **Scan**, **Enroll**, **Evaluate**.
- Python 3.12 via `uv`; Apple M1 with MPS, CPU fallback. ONNX Runtime runs on
  CPU.

Variants:
- `v0` — whole-frame DINOv2 control. Detects background shortcutting.
- `v2` — YOLO-seg bbox + mask -> DINOv2. The production path.
- `v3` — YOLO box -> SAM2 mask -> DINOv2. Evaluation comparison only.

## Setup

```bash
cd backend
uv sync --python 3.12
uv run python ../scripts/download_models.py
cd ../frontend
npm install
```

The download fetches SAM2, DINOv2, and `yolo26n-seg.pt`, then exports it to
`data/weights/yolo26n-seg.onnx` (about 250 MB total). `data/model_manifest.json`
records both YOLO files. Weights, media, and SQLite live under `data/`
(gitignored). Re-run the script after changing the YOLO model or export
settings.

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

1. YOLO26n-seg runs once and returns boxes plus masks at the original frame
   size.
2. One detection is selected: confidence first, minus a centre-distance
   penalty. Detections are rejected when they are too small or too large,
   touch the frame border, or belong to `AUTO_SCANNER_YOLO_REJECT_CLASSES`
   (default `person`).
3. The selected detection's mask is checked: non-empty, area within the YOLO
   limits, border leakage at most `AUTO_SCANNER_MASK_MAX_BORDER_FRACTION`
   (0.10), bbox IoU against the detection box at least
   `AUTO_SCANNER_MASK_MIN_BOX_IOU` (0.25), and at least
   `AUTO_SCANNER_MASK_MIN_BOX_COVERAGE` (0.50) of mask pixels inside the box.
4. The mask crop is normalized: masked-pixel channel means are equalized
   (gray-world white balance), then each channel is standardized to mean 128
   and standard deviation 50. The same step runs for gallery and live frames.
5. The normalized crop is letterboxed to 224x224 and embedded with DINOv2.

No object name from YOLO is used to identify the item. `v3` repeats steps 1-2,
prompts SAM2 with the selected box and its centre point, and applies the same
normalization and embedding to the SAM2 mask. SAM2 keeps its own predicted-IoU
threshold (`AUTO_SCANNER_SAM_MIN_PRED_IOU`).

## Enrollment

Enrollment requires both a YOLO bounding box and an accepted segmentation
mask. Frames with no detection are rejected with `no_detection`; frames where
the mask fails selection are rejected with `mask_rejected`; model failures
report `model_error`. Rejected frames write no files and no database row.
Accepted captures store the image, the mask, and a mask-tinted thumbnail.

Item name and price are editable from the Enroll tab; matching reads the
current values, so a rename or repricing applies immediately.

Near-duplicate warnings compare whole-frame `v0` vectors, so they stay
independent of detection.

Existing databases keep working. Older captures may have
`mask_rel_path=""`, `mask_found=0`, and `mask_status="not_localized"`; their
`v2` vectors are recomputed lazily from the stored image under the current
settings. No schema migration is required.

## Performance on the M1

Measured on the current stored captures with `scripts/benchmark_yolo.py`:

- YOLO26n-seg ONNX: p50 ~77 ms, p95 ~107 ms per frame (CPU ExecutionProvider).
- Warm DINOv2: ~20 ms.
- End-to-end `v2` on an accepted frame: roughly 100 ms.
- SAM2 (`v3` only): ~0.44 s per accepted frame; the first `v3` request also
  pays model load time.
- The frontend auto-scan runs every 250 ms and skips overlapping requests.

Detection coverage is the limiting factor. On the current stored captures,
YOLO detected two of three and produced a valid mask for both. A product
outside the 80 COCO classes produces no bounding box. Use
`scripts/benchmark_yolo.py --conf`, `--reject-classes`, and `--sam2` to compare
settings and the SAM2 comparison path.

## Evaluation protocol

1. **Enroll** each item with 3+ gallery captures from multiple angles in
   session `G1`.
2. Capture 5–10 **test** images in a different session `T1` with different
   angles, lighting, and distance. Never put test images in the gallery.
3. Run **Evaluate** with `v0`, `v2`, and optionally `v3` to compare the
   YOLO-seg mask against the SAM2 mask.
4. Read top-1/top-5, the 95% Wilson interval, per-item results, confusion,
   mask-accepted rate, and the failure gallery. If `v0` is much better than
   `v2`, suspect background shortcutting or mask failures.

Minimum for a weak signal: 5 items × 3 gallery × 5 test. For a credible
go/no-go: 8–12 visually similar items × 10–20 test captures each.

Evaluation uses the same stored images. `v2` and `v3` vectors are recomputed
on first use and cached with a version that includes the YOLO, mask
normalization, and SAM2 settings, so changing a threshold invalidates only
the affected vectors.

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
- Legacy columns for hand and mask metadata remain in existing databases;
  enrollment now writes real mask values for new captures.
- The ONNX export pins `imgsz=640`. Changing `AUTO_SCANNER_YOLO_IMGSZ` requires
  re-running `scripts/download_models.py` and invalidates `v2`/`v3` vectors.
- Photometric normalization was added with the YOLO-seg switch, so the 0.60
  add threshold may need retuning after an evaluation run.
