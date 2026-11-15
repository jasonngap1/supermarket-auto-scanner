import json
import time
from datetime import datetime, timezone

import numpy as np

from app.db import db
from app.evaluation.artifacts import save_failure_artifact
from app.evaluation.metrics import SampleResult, compute_metrics
from app.matching.matcher import item_topk
from app.matching.vector_index import normalize
from app.services.captures import list_test_captures
from app.services.storage import load_image_rgb


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class EvaluationRunner:
    def __init__(self, settings, embeddings):
        self.settings = settings
        self.embeddings = embeddings

    def run(self, run_id, variant_ids, item_ids, progress) -> dict:
        captures = list_test_captures(self.settings, item_ids)
        gallery = {variant: self.embeddings.gallery_entries(variant) for variant in variant_ids}
        samples: dict[str, list[SampleResult]] = {variant: [] for variant in variant_ids}
        near_duplicates = {variant: 0 for variant in variant_ids}
        total = len(captures) * len(variant_ids)
        done = 0

        for capture in captures:
            image = None
            for variant in variant_ids:
                match_ms = 0.0
                top = []
                try:
                    details = self.embeddings.get_capture_vector_details(capture, variant)
                    vector = details.vector
                    status = details.status
                    timings = dict(details.timings_ms)
                    if vector is not None:
                        entries = gallery[variant]
                        if not entries:
                            status = "no_gallery"
                        else:
                            if any(
                                entry.vectors.size
                                and float(np.max(entry.vectors @ normalize(vector))) > 0.995
                                for entry in entries
                            ):
                                near_duplicates[variant] += 1
                            match_start = time.perf_counter()
                            top = item_topk(vector, entries, k=5)
                            match_ms = (time.perf_counter() - match_start) * 1000
                except Exception:
                    status = "model_error"
                    timings = {}

                timings["match"] = match_ms
                timings["total"] = sum(timings.values())
                sample = SampleResult(
                    true_item_id=capture.item_id,
                    true_item_name=capture.item_name,
                    true_price_cents=capture.item_price_cents,
                    top=top,
                    status=status,
                    timings_ms=timings,
                )
                samples[variant].append(sample)
                correct1 = bool(top and top[0].item_id == capture.item_id)
                correct5 = any(item.item_id == capture.item_id for item in top[:5])
                artifact = None
                if status != "ok" or not correct1:
                    if image is None:
                        image = load_image_rgb(self.settings, capture.rel_path)
                    artifact = save_failure_artifact(
                        self.settings, run_id, f"{capture.id}-{variant}",
                        image, capture.item_name, top[0] if top else None,
                    )
                with db(self.settings) as connection:
                    connection.execute(
                        """
                        INSERT INTO eval_samples
                          (run_id, capture_id, variant, true_item_id, true_item_name,
                           true_price_cents, pred_item_id, pred_item_name, topk_json,
                           correct_top1, correct_top5, sample_status, timings_json,
                           rendered_rel_path, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id, capture.id, variant, capture.item_id, capture.item_name,
                            capture.item_price_cents,
                            top[0].item_id if top else None,
                            top[0].name if top else None,
                            json.dumps([item.__dict__ for item in top]),
                            int(correct1), int(correct5), status, json.dumps(timings),
                            artifact, _now(),
                        ),
                    )
                done += 1
                progress(done, total)

        metrics = {
            variant: compute_metrics(samples[variant], n_items=len({c.item_id for c in captures}))
            for variant in variant_ids
        }
        for variant in variant_ids:
            metrics[variant]["near_duplicate_count"] = near_duplicates[variant]
        return metrics
