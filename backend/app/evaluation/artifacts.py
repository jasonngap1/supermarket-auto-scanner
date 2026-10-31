from pathlib import Path

from app.domain import TopItem
from app.pipeline.annotate import annotate_frame
from app.services.storage import resolve_media_path


def save_failure_artifact(
    settings, run_id: str, key: str, image_rgb, localization,
    true_name: str, predicted: TopItem | None,
) -> str:
    label = f"true: {true_name} | pred: {predicted.name if predicted else 'none'}"
    jpeg = annotate_frame(
        image_rgb, localization, None, settings.currency_symbol, label=label
    )
    rel_path = f"runs/{run_id}/{key}.jpg"
    path: Path = resolve_media_path(settings, rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(jpeg)
    return rel_path
