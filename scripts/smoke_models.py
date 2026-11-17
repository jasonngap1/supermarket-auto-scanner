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
from app.ml.registry import ModelRegistry  # noqa: E402
from app.pipeline.yolo import (  # noqa: E402
    localize_with_sam,
    localize_with_yolo_seg,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, help="RGB image to benchmark on")
    parser.add_argument(
        "--sam2", action="store_true", help="Also run the v3 SAM2 comparison chain"
    )
    args = parser.parse_args()

    settings = Settings()
    if args.image:
        image = np.asarray(Image.open(args.image).convert("RGB"))
    else:
        image = np.full((480, 640, 3), 127, dtype=np.uint8)
    height, width = image.shape[:2]

    registry = ModelRegistry(settings)
    started = time.perf_counter()
    registry.warm()
    warm_ms = (time.perf_counter() - started) * 1000
    device = registry.status()["device"]["resolved"]
    print(f"device={device} image={width}x{height} warm={warm_ms:.0f} ms")

    started = time.perf_counter()
    vector = registry.embed(image)
    elapsed = (time.perf_counter() - started) * 1000
    print(f"dinov2: dim={vector.size} norm={np.linalg.norm(vector):.4f} {elapsed:.0f} ms")

    localization = localize_with_yolo_seg(settings, registry, image)
    mask_pixels = int(localization.mask.sum()) if localization.mask is not None else 0
    print(
        f"v2 yolo-seg: status={localization.status} bbox={localization.bbox} "
        f"mask_pixels={mask_pixels} timings={localization.timings_ms}"
    )

    if args.sam2:
        sam = localize_with_sam(
            settings,
            registry,
            image,
            localization.detections,
            localization.detection_selection,
        )
        sam_pixels = int(sam.mask.sum()) if sam.mask is not None else 0
        print(
            f"v3 sam2: status={sam.status} bbox={sam.bbox} "
            f"mask_pixels={sam_pixels} timings={sam.timings_ms}"
        )


if __name__ == "__main__":
    main()
