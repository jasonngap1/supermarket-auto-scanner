import numpy as np

from app.db import init_db
from app.evaluation.runner import EvaluationRunner
from app.matching.matcher import GalleryEntry
from app.services.captures import add_capture
from app.services.evaluation import create_run
from app.services.items import create_item
from app.services.storage import save_capture_bundle
from tests.fakes import make_settings


class FakeEmbeddings:
    def __init__(self):
        self.rows = []
        self.names = {}
        self.vectors = {}

    def gallery_entries(self, variant_id):
        grouped = {}
        for row in self.rows:
            if row["role"] != "gallery":
                continue
            grouped.setdefault(row["item_id"], []).append(row["vector"])
        return [
            GalleryEntry(item_id, self.names[item_id][0], self.names[item_id][1], np.vstack(vectors))
            for item_id, vectors in grouped.items()
        ]

    def get_capture_vector(self, capture, variant_id):
        return self.vectors[capture.id]


def seed(tmp_path):
    settings = make_settings(tmp_path)
    init_db(settings)
    fake = FakeEmbeddings()
    for idx, (name, vector) in enumerate([("Milk", [1, 0]), ("Bread", [0, 1])], start=1):
        item = create_item(settings, name, idx * 100)
        fake.names[item.id] = (name, idx * 100)
        image = np.full((16, 16, 3), 100 + idx, dtype=np.uint8)
        mask = np.ones((16, 16), dtype=bool)
        for role in ("gallery", "test"):
            artifacts = save_capture_bundle(settings, item.id, image, mask)
            capture = add_capture(
                settings,
                item_id=item.id,
                role=role,
                session_id="s",
                artifacts=artifacts,
                sha256=f"{item.id}-{role}".ljust(64, "0"),
                width=16,
                height=16,
                hand_found=True,
                mask_found=True,
                mask_status="ok",
                localization={"status": "ok", "timings_ms": {}},
            )
            fake.vectors[capture.id] = np.array(vector, dtype=np.float32)
            fake.rows.append({"item_id": item.id, "role": role, "vector": fake.vectors[capture.id]})
    return settings, fake


def test_runner_computes_perfect_accuracy(tmp_path):
    settings, fake = seed(tmp_path)
    run_id = "run-1"
    create_run(settings, run_id, {"variants": ["v2"]})
    metrics = EvaluationRunner(settings, fake).run(
        run_id, ["v2"], None, lambda done, total: None
    )
    assert metrics["v2"]["n"] == 2
    assert metrics["v2"]["top1"] == 1.0
    assert metrics["v2"]["near_duplicate_count"] == 2
