from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


BASE_MODEL_ID = "Qwen/Qwen3.5-4B"
DATASET_NAME = "TRACE+17"
DATASET_SEED = 42
STAGE_COUNT = 25
# Formal TRACE+17 decoding ceiling.
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
    """Score with this baseline's local copy of the canonical TRACE+17 evaluator."""
    from scoring import score_predictions

    return score_predictions(metric, list(predictions), list(gold))
