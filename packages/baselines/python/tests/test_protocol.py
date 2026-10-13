from pathlib import Path

import pytest
from hero_evaluation.scoring import VERSION

from hero_baselines.protocol import (
    BASE_MODEL_ID,
    DATASET_SEED,
    STAGE_COUNT,
    canonical_score,
    load_manifest,
    stream_metrics,
)


def test_pinned_protocol_identity():
    assert BASE_MODEL_ID == "Qwen/Qwen3.5-4B"
    assert DATASET_SEED == 42
    assert STAGE_COUNT == 25


def test_tracked_manifest_has_canonical_order():
    root = Path(__file__).resolve().parents[4]
    manifest = load_manifest(root / "datasets/trace-plus-17")
    assert manifest["tasks"][0]["id"] == "trace:C-STANCE"
    assert manifest["tasks"][-1]["id"] == "superni:217"


def test_canonical_evaluator_is_reused():
    gold = [{"id": "x", "prompt": "p", "references": ["A"]}]
    summary = canonical_score("strict_exact_match", [{"id": "x", "prediction": "A"}], gold)
    assert summary["score"] == 1
    assert summary["scorerVersion"] == VERSION


def test_stream_metrics_use_clipped_diagonal_to_final_forgetting():
    # Task a peaks at stage 2 (0.9) after acquisition 0.6 and ends at 0.7: F uses the diagonal, not the peak.
    # Task b gains after acquisition, which F clips at zero while signed BWT keeps the sign.
    cells = [{"stage": 1, "task": "a", "score": 0.6}, {"stage": 2, "task": "a", "score": 0.9},
             {"stage": 2, "task": "b", "score": 0.5}, {"stage": 3, "task": "a", "score": 0.7},
             {"stage": 3, "task": "b", "score": 0.8}, {"stage": 3, "task": "c", "score": 0.4}]
    metrics = stream_metrics(cells, ["a", "b", "c"])
    assert metrics["completedStages"] == 3
    assert metrics["AA"] == pytest.approx((0.7 + 0.8 + 0.4) / 3)
    assert metrics["acquisitionMeanDiagonal"] == pytest.approx((0.6 + 0.5 + 0.4) / 3)
    assert metrics["forgetting"] == pytest.approx(0.0)
    assert metrics["signedBWT"] == pytest.approx((0.1 + 0.3) / 2)
