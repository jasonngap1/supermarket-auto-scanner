# Supermarket Auto-Scanner — Model Validation Lab

Local prototype for recognizing a supermarket item held in front of a laptop
camera and retrieving its name and price. It exists to answer: are
centre-prompted segmentation and DINOv2 retrieval accurate enough for a final
scanner?

## Stack

- FastAPI + SQLite + local files; all inference runs server-side.
- SAM2 (`facebook/sam2.1-hiera-tiny`) with a **fixed centre point + centre
  box**, background corner negatives, and MediaPipe face negatives. The point
  anchors the item; face, border, and outside-ROI masks are rejected. No hand
  and no text prompt. This is the default localizer.
- SAM3-LiteText (`vil-uob/sam3-litetext-s0`) is optional: a text concept plus
  a centred search region.
- SAM2 prompted from MediaPipe hand landmarks remains a control
  (`AUTO_SCANNER_LOCALIZER=sam2`).
- Embeddings: DINOv2 (`facebook/dinov2-small`) or CLIP
  (`openai/clip-vit-base-patch32`).
- Vite vanilla JavaScript UI: **Scan**, **Enroll**, **Evaluate**.
- Python 3.12 via `uv`; Apple M1 with MPS, CPU fallback.

## Setup

```bash
cd backend
uv sync --python 3.12
uv run python ../scripts/download_models.py          # add --with-sam3, --with-clip
cd ../frontend
npm install
```

The default download is about 250 MB (SAM2 + DINOv2 + the hand and face assets).
SAM3 adds about 2.1 GB; CLIP adds about 1.1 GB. Weights and the SQLite database
live under `data/` (gitignored).

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

Hold the item on the **centre point** of the frame. The default `sam2_center`
localizer sends a positive point at the frame centre plus a centred box
(`AUTO_SCANNER_CENTER_ROI_WIDTH=0.60`, `AUTO_SCANNER_CENTER_ROI_HEIGHT=0.70`).
A mask must contain the centre point, overlap the box, and fall between
`AUTO_SCANNER_SAM2_CENTER_MIN_AREA_RATIO` (0.005) and
`AUTO_SCANNER_SAM2_CENTER_MAX_AREA_RATIO` (0.60) of the frame. The larger cap
allows an item held close to the camera.

The prompt also adds negative points at the frame corners and just outside the
item box. A visible face is added as negative points unless it covers the
centre point. Selection rejects masks that:
- touch the frame border
- extend more than
  `AUTO_SCANNER_SAM2_CENTER_ROI_MARGIN_RATIO` (3%) outside the item box
- overlap the detected face box heavily
  (`AUTO_SCANNER_SAM2_CENTER_FACE_IOU_REJECT=0.75` /
  `AUTO_SCANNER_SAM2_CENTER_FACE_COVERAGE_REJECT=0.70`)

If the centre point lands on the background or your face, expect
`mask_rejected`. Move the item so it covers the centre point.

To try the SAM3 concept localizer instead, download it with `--with-sam3` and
set `AUTO_SCANNER_LOCALIZER=sam3`. Tune `AUTO_SCANNER_SAM3_TEXT_PROMPT`
(default `product`) or lower `AUTO_SCANNER_SAM3_MIN_SCORE`. Set
`AUTO_SCANNER_LOCALIZER=sam2` to use the MediaPipe hand-prompt control.

SAM2 and SAM3 load lazily on the first scan; the first frame includes model
loading time.

V3 (CLIP) only appears when both the weights are downloaded and the backend is
started with `AUTO_SCANNER_ENABLE_CLIP=1 scripts/run_backend.sh`.

## Performance on the M1

Measured with the 960×720 test frame from `scripts/smoke_models.py` on MPS,
float32:

- SAM2 centre point + box: ~1.6 s to load, ~0.7 s for the first frame,
  **~0.42 s per frame** once warm.
- Warm DINOv2 ~20 ms; hand detector ~27 ms; face detector ~3–10 ms; retrieval
  matching ~2 ms.
- End-to-end V2 scan: roughly **0.5 s/frame**. The auto toggle can run at that
  rate.
- Enrollment runs one SAM2 inference per capture, so each accepted image costs
  roughly 0.5 s. A 10-item, 3-gallery + 10-test protocol is about 130 captures
  (~1–2 minutes of capture time).
- Optional SAM3-LiteText is ~6 s to load and **~4.8 s per frame**; it allocates
  ~2.3 GB and needs float32 because FP16 fails on MPS. Use it only when the
  text concept is worth the cost.

## Evaluation protocol

1. **Enroll** each item with 3+ gallery captures from multiple angles, held in
   the centre of the frame, in session `G1`.
2. Capture 5–10 **test** images in a *different* session `T1` with different
   angles/lighting/distance. Never put test images in the gallery.
3. Run **Evaluate** with V0/V1/V2 (and V3 if CLIP is downloaded).
4. Read top-1/top-5, the 95% confidence interval, per-item results, confusion,
   and the failure gallery. V0/V1 are controls: if V0 ≫ V2, suspect background
   shortcutting. Tune mask thresholds only on separate debug frames.

The localizer runs at enrollment time and its mask is stored with the capture.
Evaluation reuses those stored masks, so changing the localizer, prompt, or
centre ROI requires re-enrolling (or at least re-capturing) the gallery and test
sets. Keep the same localizer and prompt for both sets.

Enrollment warns when a test capture is a near-duplicate (> 0.995 cosine) of a
gallery capture for the same item, and every evaluation reports its
near-duplicate count — those images inflate accuracy.

Minimum for a weak signal: 5 items × 3 gallery × 5 test. For a credible
go/no-go: 8–12 visually similar items × 10–20 test captures each.

## Tests

```bash
cd backend && uv run pytest -v
cd frontend && npm test && npm run build
cd backend && RUN_MODEL_TESTS=1 uv run pytest -m slow -v
# With the optional SAM3-LiteText: RUN_MODEL_TESTS=1 RUN_SAM3_TESTS=1 uv run pytest -m slow -v
# With CLIP: AUTO_SCANNER_ENABLE_CLIP=1 RUN_MODEL_TESTS=1 uv run pytest -m slow -v
```

## Notes

- Evaluation is batch/offline; Scan is live inference and reports no accuracy.
- SAM2 and SAM3 produce masks, not embeddings. DINOv2 is the primary embedder;
  CLIP is a semantic baseline.
- `sam2_center` assumes the item covers the frame centre. If the centre point
  lands on the background or a face, the mask is rejected; move the item to the
  centre. Face detection is best-effort: if the face asset is missing, the app
  still runs without face rejection.
- `vil-uob/sam3-litetext-s0` is a third-party Apache-2.0 distillation of SAM3.
  The official `facebook/sam3` weights are gated and carry a separate licence;
  review the applicable terms before commercial use.
- Matching is exact cosine. HNSW is deferred behind `VectorIndex`.
- DINOv2 weight licensing must be verified before any commercial use.
- Enrollment rejects frames where the localizer finds no mask, so every stored
  capture has a valid mask. Consequently the evaluation's mask-accepted rate is
  ~100% by construction; segmentation usability shows up as enrollment
  rejections/retakes rather than in that metric.
