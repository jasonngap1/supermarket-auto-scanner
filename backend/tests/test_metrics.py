import pytest

from app.domain import TopItem
from app.evaluation.metrics import SampleResult, compute_metrics, wilson_ci


def sample(true_id, top_ids, status="ok", scores=None):
    scores = scores or [0.9 - 0.1 * i for i in range(len(top_ids))]
    return SampleResult(
        true_item_id=true_id,
        true_item_name=f"item-{true_id}",
        true_price_cents=100,
        top=[TopItem(i, f"item-{i}", 100, s) for i, s in zip(top_ids, scores)],
        status=status,
        timings_ms={"embed": 10.0, "match": 1.0, "total": 11.0},
        hand_found=status != "no_hand",
    )


def test_wilson_interval_edges():
    assert wilson_ci(0, 10)[0] == pytest.approx(0.0)
    low, high = wilson_ci(10, 10)
    assert low < 1.0 and high == pytest.approx(1.0)


def test_compute_metrics_micro_macro_and_failures():
    samples = [
        sample(1, [1, 2]),
        sample(1, [2, 1]),
        sample(2, [3, 2]),
        sample(3, [], status="no_hand"),
    ]
    metrics = compute_metrics(samples, n_items=3)
    assert metrics["n"] == 4
    assert metrics["n_items"] == 3
    assert metrics["top1"] == pytest.approx(0.25)
    assert metrics["top5"] == pytest.approx(0.75)
    assert metrics["macro_top1"] == pytest.approx((0.5 + 0.0 + 0.0) / 3)
    assert metrics["k_effective"] == 3
    assert metrics["hand_found_rate"] == pytest.approx(0.75)
    assert metrics["mask_accepted_rate"] == pytest.approx(0.75)
    assert metrics["per_item"]["item-1"]["top1"] == pytest.approx(0.5)
    assert metrics["confusion"]["item-2"]["item-3"] == 1
    assert metrics["latency_ms"]["total"]["p50"] == pytest.approx(11.0)
