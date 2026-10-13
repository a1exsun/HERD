from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


BASE_MODEL_ID = "Qwen/Qwen3.5-4B"
DATASET_NAME = "TRACE+17"
DATASET_SEED = 42
STAGE_COUNT = 25
# Formal decoding ceiling shared with the base control (see packages/baselines/base).
EVAL_MAX_NEW_TOKENS = 5120
EVAL_MAX_INPUT_TOKENS = 16384
FULL_EVALUATION_STAGES = (4, 8, 16, 25)
TRACE_PANEL_STAGES = (12, 20)
TRACE_TASK_COUNT = 8


def evaluated_task_indices(stage: int) -> list[int]:
    """Zero-based task indices scored after `stage`, matching the base control's sparse matrix."""
    if not 1 <= stage <= STAGE_COUNT:
        raise ValueError(f"Stage must be within 1..{STAGE_COUNT}.")
    if stage in FULL_EVALUATION_STAGES:
        seen = range(stage)
    elif stage in TRACE_PANEL_STAGES:
        seen = range(TRACE_TASK_COUNT)
    else:
        seen = []
    return sorted(set(seen) | {stage - 1})


def general_panel_kind(stage: int) -> str:
    """G-test panel after `stage`: the full 1,000 items at full-evaluation stages, else the 200-item probe."""
    return "full" if stage == 0 or stage in FULL_EVALUATION_STAGES else "probe"


def load_manifest(dataset_root: Path) -> dict[str, Any]:
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("name") != DATASET_NAME or manifest.get("seed") != DATASET_SEED:
        raise ValueError("Dataset manifest is not the pinned TRACE+17 seed-42 stream.")
    tasks = manifest.get("tasks", [])
    if manifest.get("taskCount") != STAGE_COUNT or len(tasks) != STAGE_COUNT:
        raise ValueError("TRACE+17 manifest must contain exactly 25 stages.")
    if [task.get("position") for task in tasks] != list(range(1, STAGE_COUNT + 1)):
        raise ValueError("TRACE+17 task positions are not the canonical 1..25 order.")
    return manifest


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_score(metric: str, predictions: Iterable[dict[str, str]], gold: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Delegate scoring to the repository's canonical TRACE/SuperNI scorer."""
    from hero_evaluation.scoring import score_predictions

    return score_predictions(metric, list(predictions), list(gold))


def stream_metrics(cells: Iterable[dict[str, Any]], order: list[str]) -> dict[str, Any]:
    """The paper's stream metrics from a sparse matrix: AA, diagonal-to-final forgetting F and signed BWT.

    F clips each diagonal-to-final loss at zero; it is not referenced to a task's best observed score.
    """
    matrix = {(c["stage"], c["task"]): c["score"] for c in cells}
    final_stage = max((stage for stage, _ in matrix), default=0)
    metrics: dict[str, Any] = {"completedStages": final_stage}
    diagonal = [matrix[(j + 1, order[j])] for j in range(final_stage) if (j + 1, order[j]) in matrix]
    if diagonal:
        metrics["acquisitionMeanDiagonal"] = sum(diagonal) / len(diagonal)
    last = len(order)
    if final_stage == last and all((last, task) in matrix for task in order):
        final = [matrix[(last, task)] for task in order]
        metrics["AA"] = sum(final) / len(final)
        earlier = list(enumerate(order[:-1], start=1))
        metrics["forgetting"] = sum(max(0.0, matrix[(j, t)] - matrix[(last, t)]) for j, t in earlier) / len(earlier)
        metrics["signedBWT"] = sum(matrix[(last, t)] - matrix[(j, t)] for j, t in earlier) / len(earlier)
    return metrics
