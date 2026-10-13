import json
from itertools import permutations
from collections import Counter

import pytest

from replay import (
    MemoryRecord, UniformReplayMemory, batch_schedule, count_serialized_tokens, pack_fixed_general, selection_key,
    serialize_record, supervised_ids, training_pool,
)
from run import stage_range, stream_metrics


class CharTokenizer:
    """One token per character; enough to exercise budgets and chat masking."""

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(text)

    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        prompt = [1, 2, 3]
        ids = prompt if add_generation_prompt else prompt + [ord(c) for c in messages[-1]["content"]]
        return {"input_ids": ids} if kwargs.get("return_dict") else ids


def item(identifier, tokens, stage=1, task="t"):
    return MemoryRecord({"id": identifier, "prompt": "p", "answer": "a"}, task, stage, tokens)


def test_serialization_matches_compact_javascript_json():
    record = {"id": "x", "prompt": "你好 \"q\"", "references": ["a", "b"]}
    assert serialize_record(record) == '{"id":"x","prompt":"你好 \\"q\\"","references":["a","b"]}\n'
    assert count_serialized_tokens(CharTokenizer(), record) == len(serialize_record(record))


def test_every_record_subset_is_equally_likely_despite_unequal_lengths(monkeypatch):
    pool = [item("A", 16000), item("B", 2), item("C", 1), item("D", 100)]
    subsets = Counter()
    for order in permutations("ABCD"):
        monkeypatch.setattr("replay.selection_key", lambda seed, identifier: order.index(identifier))
        memory = UniformReplayMemory(capacity=2, seed=42)
        memory.update(pool)
        assert len(memory.records) == memory.capacity == 2
        subsets[frozenset(r.id for r in memory.records)] += 1
    assert len(subsets) == 6 and set(subsets.values()) == {4}
    assert {key: sum(count for subset, count in subsets.items() if key in subset)
            for key in "ABCD"} == dict.fromkeys("ABCD", 12)


def test_variable_length_reservoir_matches_full_history_without_reentry():
    stream = [[item(f"s{s}-{i}", (i + 1) ** s, stage=s) for i in range(10)]
              for s in range(1, 5)]
    memory = UniformReplayMemory(capacity=12, seed=7)
    history, rejected = [], set()
    for stage in stream:
        history.extend(stage)
        memory.update(stage)
        expected = sorted(history, key=lambda r: selection_key(7, r.id))[:12]
        assert memory.records == expected
        retained = {r.id for r in memory.records}
        assert not (retained & rejected)
        rejected |= {r.id for r in history} - retained


def test_long_records_do_not_shrink_capacity_or_get_filtered(monkeypatch):
    monkeypatch.setattr("replay.selection_key", lambda seed, identifier: identifier)
    memory = UniformReplayMemory(capacity=128, seed=42)
    memory.update([item(f"b{i:03}", 100) for i in range(500)])
    memory.update([item("a-long", 16294, stage=2)])
    assert len(memory.records) == memory.capacity == 128
    assert memory.records[0].id == "a-long" and memory.records[0].tokens == 16294
    assert sum(r.tokens for r in memory.records) > 24576


def test_selection_validates_tokens_and_duplicates_before_changing_state():
    same = item("x", 1)
    memory = UniformReplayMemory(capacity=5, seed=1)
    memory.update([same, same])
    assert memory.records == [same]
    for conflicting in [MemoryRecord({"id": "x", "prompt": "other"}, "t", 1, 1), item("x", 2)]:
        with pytest.raises(ValueError, match="Conflicting"):
            memory.update([item("y", 4), conflicting])
        assert memory.records == [same]
    with pytest.raises(ValueError, match="positive"):
        memory.update([item("zero", 0)])
    for capacity in [0, -1]:
        with pytest.raises(ValueError, match="positive"):
            UniformReplayMemory(capacity=capacity, seed=1)


def test_general_reservation_is_whole_and_never_exceeded():
    general = [item("g1", 4, stage=0, task="general"), item("g2", 4, stage=0, task="general")]
    assert pack_fixed_general(general, budget=8) == general
    with pytest.raises(ValueError, match="reservation"):
        pack_fixed_general(general, budget=7)


def test_training_pool_tags_sources_and_rejects_repeated_ids():
    pool = training_pool([{"id": "c", "prompt": "p", "answer": "a"}], [item("b", 1)], [item("g", 1, task="general")])
    assert [(r["id"], r["replaySource"]) for r in pool] == [("c", "current"), ("b", "buffer"), ("g", "general")]
    with pytest.raises(ValueError, match="twice"):
        training_pool([{"id": "b", "prompt": "p", "answer": "a"}], [item("b", 1)], [])


def test_batch_schedule_is_deterministic_and_covers_each_epoch():
    first = batch_schedule(5, updates=4, batch_size=5, seed=3)
    assert first == batch_schedule(5, updates=4, batch_size=5, seed=3)
    for batch in first:
        assert sorted(batch) == list(range(5))


def test_fresh_adapter_is_seeded_before_initialization(monkeypatch, tmp_path):
    import copy
    from types import SimpleNamespace

    import torch
    from transformers import AutoModelForImageTextToText, LlamaConfig, LlamaForCausalLM
    from run import Unit

    base = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=8, intermediate_size=16,
                                       num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
    monkeypatch.setattr(AutoModelForImageTextToText, "from_pretrained", lambda *a, **kw: copy.deepcopy(base))
    unit = object.__new__(Unit)
    unit.checkpoint, unit.device, unit.tokenizer = tmp_path, "cpu", SimpleNamespace(eos_token_id=2)
    unit.plan = {"training": {"seed": 42, "lora": {"rank": 2, "alpha": 4, "dropout": 0.0}}}

    def fresh(incoming_seed):
        torch.manual_seed(incoming_seed)
        model = unit.model(None, trainable=True)
        return {name: p.detach().clone() for name, p in model.named_parameters() if "lora_A" in name}

    first, second = fresh(11), fresh(22)
    assert first and first.keys() == second.keys()
    assert all(torch.equal(first[name], second[name]) for name in first)
    unit.plan["training"]["seed"] = 43
    other = fresh(11)
    assert any(not torch.equal(first[name], other[name]) for name in first)


def test_replay_training_resume_matches_uninterrupted_updates():
    import copy

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM
    from replay import train_stage

    torch.manual_seed(42)
    base = LlamaForCausalLM(LlamaConfig(vocab_size=128, hidden_size=8, intermediate_size=16,
                                       num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2))
    initial = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                                             target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    tokenizer = CharTokenizer()
    tokenizer.pad_token_id = 0
    rows = training_pool([{"id": "c", "prompt": "p", "answer": "ab"}],
                         [item("b", 10000)], [item("g", 10, task="general")])
    kwargs = dict(updates=3, batch_size=2, learning_rate=1e-3, gradient_clip=1.0,
                  seed=42, max_length=16, log=lambda row: None)
    continuous = copy.deepcopy(initial)
    expected = train_stage(continuous, tokenizer, rows, **kwargs)
    interrupted = copy.deepcopy(initial)
    saved = {}

    class Interrupted(Exception):
        pass

    def checkpoint(progress, optimizer):
        saved.update(progress=progress, optimizer=copy.deepcopy(optimizer.state_dict()),
                     weights=copy.deepcopy(interrupted.state_dict()))
        raise Interrupted

    with pytest.raises(Interrupted):
        train_stage(interrupted, tokenizer, rows, **kwargs, checkpoint_every=1, checkpoint=checkpoint)
    resumed = copy.deepcopy(initial)
    resumed.load_state_dict(saved["weights"])
    actual = train_stage(resumed, tokenizer, rows, **kwargs,
                         resume=saved["progress"], optimizer_state=saved["optimizer"])
    assert actual.mean_loss == expected.mean_loss and actual.target_tokens == expected.target_tokens
    assert all(torch.equal(value, resumed.state_dict()[name]) for name, value in continuous.state_dict().items())


def test_supervised_ids_mask_the_prompt_and_refuse_truncation():
    ids, labels = supervised_ids(CharTokenizer(), {"id": "x", "prompt": "p", "answer": "ok"}, max_length=16)
    assert labels[:3] == [-100, -100, -100] and labels[3:] == [ord("o"), ord("k")]
    with pytest.raises(ValueError, match="exceeds"):
        supervised_ids(CharTokenizer(), {"id": "x", "prompt": "p", "answer": "ok"}, max_length=4)


def test_stream_metrics_follow_the_paper_definitions():
    order = ["a", "b", "c"]
    cells = [{"stage": 1, "task": "a", "score": 0.8}, {"stage": 2, "task": "b", "score": 0.5},
             {"stage": 3, "task": "a", "score": 0.6}, {"stage": 3, "task": "b", "score": 0.7},
             {"stage": 3, "task": "c", "score": 0.4}]
    metrics = stream_metrics(cells, order)
    assert metrics["AA"] == pytest.approx((0.6 + 0.7 + 0.4) / 3)
    # Diagonal-to-final forgetting clamps gains at zero; signed BWT keeps them.
    assert metrics["forgetting"] == pytest.approx((0.2 + 0.0) / 2)
    assert metrics["signedBWT"] == pytest.approx((-0.2 + 0.2) / 2)
    assert metrics["acquisitionMeanDiagonal"] == pytest.approx((0.8 + 0.5 + 0.4) / 3)
    partial = stream_metrics(cells[:2], order)
    assert "AA" not in partial and partial["completedStages"] == 2


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
