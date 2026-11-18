import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import bootstrap  # noqa: E402,F401
from app.config import Settings  # noqa: E402
from app.ml.registry import ModelRegistry  # noqa: E402
from app.pipeline.yolo import localize_with_yolo_seg  # noqa: E402
from app.services.captures import (  # noqa: E402
    list_gallery_captures,
    list_test_captures,
)
from app.services.embeddings import EmbeddingService  # noqa: E402
from app.services.storage import load_image_rgb  # noqa: E402

EMBED_VARIANTS = ("v0", "v2")


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Also compute v0/v2 embeddings for every stored capture",
    )
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--reject-classes", default=None)
    args = parser.parse_args()

    overrides = {}
    if args.conf is not None:
        overrides["yolo_conf"] = args.conf
    if args.reject_classes is not None:
        overrides["yolo_reject_classes"] = args.reject_classes
    settings = Settings(
        data_dir=args.data_dir or ROOT / "data",
        warm_models=False,
        **overrides,
    )
    registry = ModelRegistry(settings)
    registry.warm()

    rows = [
        (f"gallery:{capture.id}", capture, name)
        for capture, name, _ in list_gallery_captures(settings)
    ]
    rows.extend(
        (f"test:{capture.id}", capture, name)
        for capture, name in list_test_captures(settings)
    )
    if not rows:
        raise SystemExit("No stored captures found")

    first_image = load_image_rgb(settings, rows[0][1].rel_path)
    localize_with_yolo_seg(settings, registry, first_image)

    latencies: dict[str, list[float]] = {"yolo": []}
    accepted = 0
    for name, capture, _label in rows:
        image = load_image_rgb(settings, capture.rel_path)
        localization = localize_with_yolo_seg(settings, registry, image)
        accepted += localization.status == "ok"
        if "yolo" in localization.timings_ms:
            latencies["yolo"].append(localization.timings_ms["yolo"])
        selected = localization.detection_selection.selected
        selected_text = (
            "-"
            if selected is None
            else f"{selected.class_name} {selected.confidence:.2f} "
            f"{tuple(round(v) for v in selected.bbox)}"
        )
        mask_pixels = int(localization.mask.sum()) if localization.mask is not None else 0
        print(
            f"{name} v2 {localization.status:12s} selected={selected_text} "
            f"mask_pixels={mask_pixels} timings="
            f"{ {k: round(v, 1) for k, v in localization.timings_ms.items()} }"
        )

    print()
    print(f"v2 accepted {accepted}/{len(rows)}")
    for stage, values in latencies.items():
        p50 = statistics.median(values) if values else 0.0
        print(f"{stage}: p50 {p50:.1f} ms p95 {percentile(values, 0.95):.1f} ms")

    if args.embed:
        service = EmbeddingService(settings, registry)
        print()
        for variant in EMBED_VARIANTS:
            ok = 0
            statuses: dict[str, int] = {}
            for _name, capture, _label in rows:
                result = service.get_capture_vector_details(capture, variant)
                ok += result.vector is not None
                statuses[result.status] = statuses.get(result.status, 0) + 1
            print(f"{variant}: vectors {ok}/{len(rows)} statuses {statuses}")

    print()
    print(
        "Accuracy is not computed: the database needs held-out test captures "
        "from a different session. Use the Evaluate tab after capturing them."
    )


if __name__ == "__main__":
    main()
