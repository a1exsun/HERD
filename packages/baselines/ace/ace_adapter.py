"""TRACE+17 records as upstream ACE samples, and the settings of the upstream offline trainer."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from protocol import BASE_MODEL_ID, canonical_score

UPSTREAM = Path(__file__).resolve().parent / "upstream"


def activate_upstream() -> None:
    """Put `upstream/` on the import path, as its own entry points expect (`from ace import ACE`)."""
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))


class TraceTarget(str):
    """The reference answer that ACE shows its Reflector, carrying the complete record for the scorer.

    Upstream calls `answer_is_correct(predicted, target)` with the sample's target only, while TRACE+17
    metrics also read other references, the source prompt (SARI) or metadata. As a `str` the target is
    exactly the first reference everywhere upstream formats, compares or saves it.
    """

    record: dict[str, Any]

    def __new__(cls, record: Mapping[str, Any]) -> "TraceTarget":
        references = list(record.get("references") or [])
        answer = record.get("answer") or (references[0] if references else None)
        if not isinstance(answer, str) or not answer:
            raise ValueError(f"Training record {record.get('id')} has no target.")
        target = super().__new__(cls, answer)
        target.record = dict(record)
        return target

    def __reduce__(self):
        return TraceTarget, (self.record,)


def to_ace_sample(record: Mapping[str, Any]) -> dict[str, Any]:
    """Upstream's sample shape. The whole TRACE+17 prompt is the question and the context is empty, as
    upstream's Formula processor does for prompts it does not split."""
    return {"question": str(record["prompt"]), "context": "", "target": TraceTarget(record)}


class TraceDataProcessor:
    """Upstream's DataProcessor interface over the canonical TRACE+17 scorer.

    ACE needs a yes/no answer check; an answer is correct when its canonical item score is full credit.
    """

    def __init__(self, metric: str, full_credit: float = 1.0):
        self.metric, self.full_credit = metric, full_credit

    def item_score(self, predicted: str, target: TraceTarget) -> float:
        record = target.record
        return float(canonical_score(self.metric, [{"id": str(record["id"]), "prediction": predicted}],
                                     [record])["score"])

    def answer_is_correct(self, predicted: str, ground_truth: TraceTarget) -> bool:
        return self.item_score(predicted, ground_truth) + 1e-12 >= self.full_credit

    def evaluate_accuracy(self, out: Sequence[str], target: Sequence[TraceTarget]) -> float:
        if len(out) != len(target):
            raise ValueError("Predictions and targets differ in length.")
        return sum(self.item_score(p, t) for p, t in zip(out, target)) / max(1, len(out))


# Every call sends at most 4,096 new tokens (upstream ACE's default); the server's context of
# 16,384 + 4,096 tokens therefore rejects any prompt above the shared 16,384-token input ceiling.
ROLE_MAX_TOKENS = 4096
MAX_MODEL_LEN = 16384 + ROLE_MAX_TOKENS


def ace_config(*, save_dir: str, task_name: str) -> dict[str, Any]:
    """Upstream offline-mode settings, all the defaults of `eval/finance/run.py` except the protocol's
    `continue_on_llm_error`."""
    return {
        "task_name": task_name,
        "save_dir": save_dir,
        "num_epochs": 1,
        "max_num_rounds": 3,
        "curator_frequency": 1,
        # Every 100 steps the playbook is scored on the task's validation split; the best one is selected.
        "eval_steps": 100,
        "save_steps": 50,
        "playbook_token_budget": 80000,
        "test_workers": 20,
        "json_mode": False,
        "no_ground_truth": False,
        "use_bulletpoint_analyzer": False,
        "continue_on_llm_error": True,
    }


def ace_system(playbook: str | None = None, next_global_id: int | None = None):
    """Upstream `ACE` (the sequential trainer of the paper, batch size 1) with every role on Qwen3.5-4B."""
    activate_upstream()
    from ace import ACE

    ace = ACE(api_provider="local", generator_model=BASE_MODEL_ID, reflector_model=BASE_MODEL_ID,
              curator_model=BASE_MODEL_ID, max_tokens=ROLE_MAX_TOKENS, initial_playbook=playbook)
    if next_global_id is not None:
        ace.next_global_id = next_global_id
    return ace
