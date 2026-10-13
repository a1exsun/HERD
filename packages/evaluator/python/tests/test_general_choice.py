"""General-choice scoring contract, including every standalone scorer snapshot."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SCORERS = sorted((ROOT / "packages/baselines").glob("*/scoring.py")) + [
    ROOT / "packages/hero-drift-constrained-opd/stream/scoring.py",
    ROOT / "packages/evaluator/python/src/hero_evaluation/scoring.py",
]


@pytest.fixture(params=SCORERS, ids=lambda path: str(path.relative_to(ROOT)))
def scorer(request):
    spec = importlib.util.spec_from_file_location("choice_scorer", request.param)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("prediction, expected", [
    ("  **b**  ", "B"),
    ("**B. The selected alternative.**\n\nExplanation follows.", "B"),
    ("Reasoning first.\n\n**Answer:**\nD. All three statements.", "D"),
    ("The correct choice is:\n\n**A. Trade networks**", "A"),
    ("The answer is (b).", "B"),
    ("The answer is B because the event caused the outcome.", "B"),
    ("Answer: A\nAfter checking again, final answer: C", "C"),
    ("Reasoning first.\n\n**D. The chosen continuation.**", "D"),
    ("Based on the context, several possibilities exist.", None),
    ("A. First alternative\nB. Second alternative", None),
    ("Answer: A or B", None),
    ("Answer: A/B", None),
    ("Answer: A, B", None),
    ("B. Initially selected\nFinal answer: E", None),
    ("The answer is not clear.", None),
    ("The answer is a plausible continuation.", None),
    ("  ", None),
])
def test_general_choice_contract(scorer, prediction, expected):
    assert scorer.VERSION == "hero-task-scores-v3"
    assert scorer.extract_general_choice(prediction, "mmlu") == expected
    # Extraction never consults the gold: one response can match at most one label.
    for source in ("mmlu", "hellaswag"):
        for reference in "ABCD":
            assert scorer.score_one("leading_choice_accuracy", prediction, reference, {"source": source}) == float(expected == reference)


def test_two_option_labels_and_trace_boundary(scorer):
    assert scorer.extract_general_choice("Answer: C", "xcopa") is None
    assert scorer.score_one("leading_choice_accuracy", "Answer: **b**", "B", {"source": "xcopa"}) == 1
    assert scorer.score_one("leading_choice_accuracy", " B", "B", {"source": "trace"}) == 0
    assert scorer.score_one("leading_choice_accuracy", "b", "B", {"source": "trace"}) == 0
    assert scorer.score_one("leading_choice_accuracy", "B explanation", "B", {"source": "trace"}) == 1
