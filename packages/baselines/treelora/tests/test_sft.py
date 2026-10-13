import pytest

from run import stage_range, stream_metrics
from sft import batch_schedule, supervised_token_ids


class FakeTokenizer:
    """Mimics transformers 5: tokenized chat templates return a mapping with input_ids."""

    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        assert kwargs.get("return_dict") is True
        prompt = [1, 2, 3]
        ids = prompt if add_generation_prompt else prompt + [ord(char) for char in messages[-1]["content"]]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_supervised_template_masks_only_prompt_tokens():
    ids, labels = supervised_token_ids(FakeTokenizer(), {"id": "x", "prompt": "p", "answer": "ok"})
    assert ids == [1, 2, 3, ord("o"), ord("k")]
    assert labels[:3] == [-100, -100, -100]
    assert labels[3:] == [ord("o"), ord("k")]


def test_supervised_template_refuses_silent_truncation():
    with pytest.raises(ValueError, match="silent truncation"):
        supervised_token_ids(FakeTokenizer(), {"id": "x", "prompt": "p", "answer": "ok"}, max_length=4)


def test_batch_schedule_is_deterministic_and_covers_each_epoch():
    schedule = batch_schedule(4, updates=3, batch_size=4, seed=7)
    assert schedule == batch_schedule(4, updates=3, batch_size=4, seed=7)
    assert all(sorted(batch) == [0, 1, 2, 3] for batch in schedule)


def test_stream_metrics_follow_the_paper_definitions():
    order = ["a", "b"]
    cells = [{"stage": 1, "task": "a", "score": 0.8}, {"stage": 2, "task": "a", "score": 0.5},
             {"stage": 2, "task": "b", "score": 0.6}]
    metrics = stream_metrics(cells, order)
    assert metrics["AA"] == pytest.approx(0.55)
    assert metrics["forgetting"] == pytest.approx(0.3)
    assert metrics["signedBWT"] == pytest.approx(-0.3)


def test_forgetting_reads_the_final_row_on_the_diagonal_items():
    order = ["a", "b"]
    cells = [{"stage": 1, "task": "a", "score": 1.0}, {"stage": 2, "task": "a", "score": 0.5},
             {"stage": 2, "task": "b", "score": 0.6}]
    items = {(1, "a"): {"x": 1.0, "y": 1.0}, (2, "a"): {"x": 1.0, "y": 0.0, "z": 0.5, "w": 0.0},
             (2, "b"): {"x": 0.6}}
    metrics = stream_metrics(cells, order, items)
    assert metrics["AA"] == pytest.approx(0.55)  # the complete final row
    assert metrics["forgetting"] == pytest.approx(0.5)  # 1.0 against 0.5 on items x and y
    assert metrics["signedBWT"] == pytest.approx(-0.5)


def test_units_run_in_stage_order_and_groups_need_one_stage(tmp_path):
    import argparse

    from run import run_units

    calls = []
    args = argparse.Namespace(stage=[3, 4], output=tmp_path / "output", groups=None)
    run_units(args, (lambda a: calls.append(("train", a.stage)), lambda a: calls.append(("evaluate", a.stage))))
    assert calls == [("train", 3), ("evaluate", 3), ("train", 4), ("evaluate", 4)]
    with pytest.raises(ValueError):
        run_units(argparse.Namespace(stage=[3, 4], output=tmp_path, groups="general:x"), ())
