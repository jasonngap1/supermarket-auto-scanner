import json
import threading
import uuid
from datetime import datetime, timezone

from app.db import db
from app.evaluation.runner import EvaluationRunner
from app.pipeline.variants import VARIANTS
from app.services.captures import list_test_captures
from app.services.items import list_items


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_run(settings, run_id, request, progress_total=0):
    with db(settings) as connection:
        connection.execute(
            """
            INSERT INTO eval_runs
              (id, created_at, status, request_json, progress_done, progress_total)
            VALUES (?, ?, 'queued', ?, 0, ?)
            """,
            (run_id, _now(), json.dumps(request), progress_total),
        )


def recover_stale_runs(settings):
    with db(settings) as connection:
        connection.execute(
            """
            UPDATE eval_runs SET status = 'failed', error = 'interrupted by restart'
            WHERE status IN ('queued', 'running')
            """
        )


class EvaluationService:
    def __init__(self, settings, embeddings):
        self.settings = settings
        self.embeddings = embeddings
        self._thread = None

    def _validate(self, variant_ids, item_ids):
        if not variant_ids or any(variant not in VARIANTS for variant in variant_ids):
            raise ValueError("Select at least one valid variant")
        if "v3" in variant_ids and not self.settings.enable_clip:
            raise ValueError("V3 requires CLIP (AUTO_SCANNER_ENABLE_CLIP=1)")
        items = list_items(self.settings)
        if item_ids:
            items = [item for item in items if item.id in set(item_ids)]
        usable = [
            item for item in items
            if item.gallery_count >= 1 and item.test_count >= 1
        ]
        if len(usable) < 2:
            raise ValueError("Need at least 2 items with both gallery and test captures")
        if len(list_test_captures(self.settings, item_ids)) == 0:
            raise ValueError("No test captures found")
        return usable

    def start(self, variant_ids, item_ids=None, background=True) -> str:
        self._validate(variant_ids, item_ids)
        if self._thread and self._thread.is_alive():
            raise RuntimeError("Evaluation already running")
        run_id = uuid.uuid4().hex
        request = {"variants": variant_ids, "item_ids": item_ids}
        total = len(list_test_captures(self.settings, item_ids)) * len(variant_ids)
        create_run(self.settings, run_id, request, progress_total=total)
        if background:
            self._thread = threading.Thread(
                target=self._run, args=(run_id, variant_ids, item_ids), daemon=True
            )
            self._thread.start()
        else:
            self._run(run_id, variant_ids, item_ids)
        return run_id

    def _run(self, run_id, variant_ids, item_ids):
        with db(self.settings) as connection:
            connection.execute(
                "UPDATE eval_runs SET status='running', started_at=? WHERE id=?",
                (_now(), run_id),
            )

        def progress(done, total):
            with db(self.settings) as connection:
                connection.execute(
                    "UPDATE eval_runs SET progress_done=?, progress_total=? WHERE id=?",
                    (done, total, run_id),
                )

        try:
            metrics = EvaluationRunner(self.settings, self.embeddings).run(
                run_id, variant_ids, item_ids, progress
            )
            with db(self.settings) as connection:
                connection.execute(
                    """
                    UPDATE eval_runs SET status='completed', finished_at=?,
                      metrics_json=?, progress_done=progress_total WHERE id=?
                    """,
                    (_now(), json.dumps(metrics), run_id),
                )
        except Exception as exc:
            with db(self.settings) as connection:
                connection.execute(
                    "UPDATE eval_runs SET status='failed', error=?, finished_at=? WHERE id=?",
                    (str(exc), _now(), run_id),
                )

    def get(self, run_id):
        with db(self.settings) as connection:
            row = connection.execute("SELECT * FROM eval_runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                return None
            failures = connection.execute(
                """
                SELECT * FROM eval_samples
                WHERE run_id=? AND (correct_top1=0 OR sample_status!='ok')
                ORDER BY id LIMIT 50
                """,
                (run_id,),
            ).fetchall()
        result = dict(row)
        result["metrics"] = json.loads(result.pop("metrics_json") or "null")
        result["request"] = json.loads(result.pop("request_json"))
        result["failures"] = [dict(f) for f in failures]
        return result

    def list(self):
        with db(self.settings) as connection:
            rows = connection.execute(
                "SELECT * FROM eval_runs ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]
