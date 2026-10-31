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
from app.pipeline.embedders import ClipEmbedder, DinoV2Embedder  # noqa: E402
from app.pipeline.hand import HandDetector  # noqa: E402
from app.pipeline.prompts import build_center_prompt, build_sam3_prompt  # noqa: E402
from app.pipeline.sam2 import Sam2Segmenter  # noqa: E402
from app.pipeline.sam3 import Sam3LiteSegmenter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, help="RGB image to benchmark on")
    parser.add_argument("--localizer", choices=["sam2", "sam2_center", "sam3"])
    parser.add_argument("--with-clip", action="store_true")
    args = parser.parse_args()

    settings = Settings()
    localizer = args.localizer or settings.localizer
    device = resolve_device(settings.device)
    if args.image:
        image = np.asarray(Image.open(args.image).convert("RGB"))
    else:
        image = np.full((480, 640, 3), 127, dtype=np.uint8)
    height, width = image.shape[:2]
    print(f"device={device} localizer={localizer} image={width}x{height}")

    if settings.hand_asset.exists():
        started = time.perf_counter()
        detector = HandDetector(settings.hand_asset)
        result = detector.detect(image)
        elapsed = (time.perf_counter() - started) * 1000
        print(f"hand: loaded, detect={result is not None}, {elapsed:.0f} ms")
    else:
        print(f"hand: asset missing at {settings.hand_asset}")

    started = time.perf_counter()
    dino = DinoV2Embedder(settings.dinov2_model_id, device)
    vector = dino.embed(image)
    elapsed = (time.perf_counter() - started) * 1000
    print(f"dinov2: dim={vector.size} norm={np.linalg.norm(vector):.4f} {elapsed:.0f} ms")

    if localizer == "sam3":
        started = time.perf_counter()
        segmenter = Sam3LiteSegmenter(
            settings.sam3_model_id,
            device,
            min_score=settings.sam3_min_score,
            max_candidates=settings.sam3_max_candidates,
        )
        load_ms = (time.perf_counter() - started) * 1000
        prompt = build_sam3_prompt(
            width,
            height,
            settings.sam3_text_prompt,
            include_box=settings.sam3_prompt_box,
            roi_width=settings.center_roi_width,
            roi_height=settings.center_roi_height,
        )
        started = time.perf_counter()
        masks = segmenter.segment(image, prompt)
        first_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        masks = segmenter.segment(image, prompt)
        steady_ms = (time.perf_counter() - started) * 1000
        scores = [round(mask.pred_iou, 3) for mask in masks]
        print(
            f"sam3: load={load_ms:.0f} ms first={first_ms:.0f} ms "
            f"steady={steady_ms:.0f} ms candidates={len(masks)} scores={scores}"
        )
    else:
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
            f"{localizer}: load={load_ms:.0f} ms first={first_ms:.0f} ms "
            f"steady={steady_ms:.0f} ms candidates={len(masks)}"
        )

    if args.with_clip:
        started = time.perf_counter()
        clip = ClipEmbedder(settings.clip_model_id, device)
        vector = clip.embed(image)
        elapsed = (time.perf_counter() - started) * 1000
        print(f"clip: dim={vector.size} norm={np.linalg.norm(vector):.4f} {elapsed:.0f} ms")


if __name__ == "__main__":
    main()
