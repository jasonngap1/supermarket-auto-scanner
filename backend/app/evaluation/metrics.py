import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from app.domain import TopItem


@dataclass(frozen=True)
class SampleResult:
    true_item_id: int
    true_item_name: str
    true_price_cents: int
    top: list[TopItem]
    status: str
    timings_ms: dict[str, float]


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else 0.0


def compute_metrics(samples: list[SampleResult], n_items: int) -> dict:
    n = len(samples)
    correct1 = [bool(s.top and s.top[0].item_id == s.true_item_id) for s in samples]
    correct5 = [
        bool(any(item.item_id == s.true_item_id for item in s.top[:5])) for s in samples
    ]

    per_item: dict[str, dict] = defaultdict(lambda: {"n": 0, "correct_top1": 0, "correct_top5": 0})
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    margins: list[float] = []

    for sample, ok1, ok5 in zip(samples, correct1, correct5):
        bucket = per_item[sample.true_item_name]
        bucket["n"] += 1
        bucket["correct_top1"] += int(ok1)
        bucket["correct_top5"] += int(ok5)
        if sample.top:
            confusion[sample.true_item_name][sample.top[0].name] += 1
        true_score = next(
            (item.score for item in sample.top if item.item_id == sample.true_item_id), 0.0
        )
        best_other = max(
            (item.score for item in sample.top if item.item_id != sample.true_item_id),
            default=0.0,
        )
        margins.append(true_score - best_other)

    per_item_out = {
        name: {
            "n": bucket["n"],
            "top1": bucket["correct_top1"] / bucket["n"],
            "top5": bucket["correct_top5"] / bucket["n"],
        }
        for name, bucket in per_item.items()
    }
    macro = (
        sum(values["top1"] for values in per_item_out.values()) / len(per_item_out)
        if per_item_out
        else 0.0
    )
    total1 = sum(correct1)
    stages = sorted({key for s in samples for key in s.timings_ms})
    latency = {
        stage: {
            "p50": percentile([s.timings_ms.get(stage, 0.0) for s in samples], 50),
            "p95": percentile([s.timings_ms.get(stage, 0.0) for s in samples], 95),
        }
        for stage in stages
    }
    mask_accepted_rate = (
        sum(
            1 for sample in samples
            if sample.status not in {"no_detection", "mask_rejected", "model_error"}
        ) / n if n else 0.0
    )

    return {
        "n": n,
        "n_items": n_items,
        "top1": total1 / n if n else 0.0,
        "top5": sum(correct5) / n if n else 0.0,
        "macro_top1": macro,
        "top1_ci": wilson_ci(total1, n),
        "k_effective": min(5, n_items),
        "mask_accepted_rate": mask_accepted_rate,
        "per_item": per_item_out,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "margin_p10": percentile(margins, 10),
        "margin_p50": percentile(margins, 50),
        "latency_ms": latency,
    }
