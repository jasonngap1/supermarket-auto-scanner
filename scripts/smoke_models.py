import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import bootstrap  # noqa: E402,F401
from app.config import Settings  # noqa: E402
from app.ml.devices import resolve_device  # noqa: E402
from app.pipeline.embedders import DinoV2Embedder  # noqa: E402
from app.pipeline.face import FaceDetector  # noqa: E402
from app.pipeline.preprocess import letterbox  # noqa: E402
from app.pipeline.prompts import build_center_prompt  # noqa: E402
from app.pipeline.sam2 import Sam2Segmenter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, help="RGB image to benchmark on")
    args = parser.parse_args()

    settings = Settings()
    device = resolve_device(settings.device)
    if args.image:
        image = np.asarray(Image.open(args.image).convert("RGB"))
    else:
        image = np.full((480, 640, 3), 127, dtype=np.uint8)
    height, width = image.shape[:2]
    print(f"device={device} image={width}x{height}")

    if settings.face_asset.exists():
        started = time.perf_counter()
        detector = FaceDetector(settings.face_asset)
        result = detector.detect(image)
        elapsed = (time.perf_counter() - started) * 1000
        print(f"face: detected={result is not None} {elapsed:.0f} ms")
    else:
        print(f"face: asset missing at {settings.face_asset}")

    started = time.perf_counter()
    dino = DinoV2Embedder(settings.dinov2_model_id, device)
    vector = dino.embed(letterbox(image, settings.input_size))
    elapsed = (time.perf_counter() - started) * 1000
    print(f"dinov2: dim={vector.size} norm={np.linalg.norm(vector):.4f} {elapsed:.0f} ms")

    started = time.perf_counter()
    segmenter = Sam2Segmenter(settings.sam2_model_id, device)
    load_ms = (time.perf_counter() - started) * 1000
    prompt = build_center_prompt(
        width,
        height,
        roi_width=settings.center_roi_width,
        roi_height=settings.center_roi_height,
    )
    started = time.perf_counter()
    masks = segmenter.segment(image, prompt)
    first_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    masks = segmenter.segment(image, prompt)
    steady_ms = (time.perf_counter() - started) * 1000
    print(
        f"sam2-center: load={load_ms:.0f} ms first={first_ms:.0f} ms "
        f"steady={steady_ms:.0f} ms candidates={len(masks)}"
    )


if __name__ == "__main__":
    main()
