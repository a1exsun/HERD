import pytest

from run import stream_metrics
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
