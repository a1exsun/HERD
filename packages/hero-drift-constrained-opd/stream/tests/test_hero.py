"""CPU checks of the HERO stream primitives on a tiny random Qwen3.5 (path in HERO_TINY_QWEN)."""
import os
import random
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hero  # noqa: E402

TINY = os.environ.get("HERO_TINY_QWEN")
pytestmark = pytest.mark.skipif(not TINY, reason="set HERO_TINY_QWEN to a tiny Qwen3.5 checkpoint")


def tiny(init_lora_weights=False):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText

    torch.manual_seed(0)
    base = AutoModelForImageTextToText.from_pretrained(TINY, dtype=torch.float32)
    targets = sorted(n for n, m in base.named_modules() if isinstance(m, torch.nn.Linear)
                     and not any(p in n for p in ("visual", "lm_head")))
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=targets, init_lora_weights=init_lora_weights))
    return model


def reference_kl(model, pairs):
    """Unchunked oracle: full logits from ordinary model calls."""
    values = []
    for pair in pairs:
        def log_probs(prompt, use_adapter):
            ids = torch.tensor([prompt + pair.completion[:-1]])
            if use_adapter:
                logits = model(input_ids=ids).logits
            else:
                with model.disable_adapter():
                    logits = model(input_ids=ids).logits
            return F.log_softmax(logits[0, len(prompt) - 1:].float(), -1)
        s, t = log_probs(pair.student, True), log_probs(pair.teacher, False).detach()
        kl = (t.exp() * (t - s)).sum(-1)
        values.append(kl.mean())
    return torch.stack(values)


def test_divergence_matches_unchunked_oracle():
    model = tiny()
    head = hero.backbone_and_head(model)[1].weight.detach().float()
    pairs = [hero.Pair([5, 6, 7, 8], [9, 10, 11], teacher=[1, 2, 5, 6, 7, 8]),
             hero.Pair([3, 4], [12, 13, 14, 15, 16], teacher=[3, 4]),
             hero.Pair([7, 8, 9], [20], teacher=[2, 7, 8, 9])]
    divergence = hero.Divergence(head, token_budget=12, chunk=2)
    divergence.attach_teacher(model, pairs)
    model.zero_grad()
    values = divergence(model, pairs, 0.5)
    ours = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    model.zero_grad()
    expected = reference_kl(model, pairs)
    (0.5 * expected.mean()).backward()
    assert torch.allclose(torch.tensor(values), expected.detach(), atol=1e-5)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert torch.allclose(ours[name], parameter.grad, atol=1e-5, rtol=1e-4), name


def test_drift_is_zero_at_stage_start_and_positive_after_change():
    head_of = lambda model: hero.backbone_and_head(model)[1].weight.detach().float()
    fresh = tiny(init_lora_weights=True)  # B = 0: the student equals the frozen checkpoint
    pairs = [hero.Pair([5, 6, 7], [8, 9, 10])]
    hero.anchor_targets(fresh, pairs)
    assert abs(hero.measure_drift(fresh, hero.Divergence(head_of(fresh)), pairs)) < 1e-6
    moved = tiny()
    pairs = [hero.Pair([5, 6, 7], [8, 9, 10])]
    hero.anchor_targets(moved, pairs)
    assert hero.measure_drift(moved, hero.Divergence(head_of(moved)), pairs) > 0


def test_harness_routing_and_budget():
    harness = hero.Harness().with_skill({"id": "a", "promptPrefix": "Solve the", "instructions": "x"})
    harness = harness.with_skill({"id": "b", "promptPrefix": "Solve the following math", "instructions": "y"})
    assert harness.route("solve  the following MATH problem")["id"] == "b"
    assert harness.route("Solve the riddle")["id"] == "a"
    assert harness.route("Translate") is None
    replaced = harness.with_skill({"id": "c", "promptPrefix": "solve the", "instructions": "z"})
    assert sorted(s["id"] for s in replaced.skills) == ["b", "c"]
    kept = hero.top_budget([{"id": "1", "tokens": 5}, {"id": "2", "tokens": 7}, {"id": "3", "tokens": 4}],
                           {"1": 1.0, "2": 3.0, "3": 2.0}, 11)
    assert [r["id"] for r in kept] == ["2", "3"]


def test_group_drift_takes_the_worst_group():
    class Fixed:  # divergence stub returning fixed per-probe values
        def __call__(self, model, pairs, weight):
            return [p.target for p in pairs]
    probes = [hero.Pair([1], [2], target=v, group=g) for v, g in
              [(0.01, "a"), (0.02, "a"), (0.01, "a"), (0.30, "b"), (0.10, "b"), (0.0, "c")]]
    value, worst = hero.group_drift(Fixed(), None, probes)
    assert abs(value - 0.20) < 1e-9 and [p.group for p in worst] == ["b", "b"]
    assert hero.group_drift(Fixed(), None, []) == (0.0, [])


def test_skill_proposal_helpers():
    class Words:  # whitespace tokenizer stub
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": text.split()}
        def decode(self, ids):
            return " ".join(ids)
    tok = Words()
    records = [{"prompt": "Do X: a1 a2", "answer": "Recommendation to adopt"},
               {"prompt": "Do X: b1", "answer": "recommendation TO x"},
               {"prompt": "Do X: c1 c2 c3", "answer": "Summary of the meeting"}]
    assert [r["answer"][:3] for r in hero.distinct_first(records)] == ["Rec", "Sum", "rec"]
    assert hero.clip(tok, "a b c d e f g h i", 6, 2) == "a b c d ... h i" and hero.clip(tok, "a b", 6, 2) == "a b"
    prompt = hero.proposal_prompt(records[:2], [(records[2], "wrong guess")], tok, input_tokens=9, output_tokens=9)
    assert '<candidate n="2">' in prompt and "<model_output>\nwrong guess\n</model_output>" in prompt
    assert "{candidates}" not in prompt and "{mistakes}" not in prompt
    assert "<mistake>" not in hero.proposal_prompt(records, [], tok, input_tokens=9, output_tokens=9)
    reply = "Sure, here is the skill.\n**Format:** one word.\nRules:\n- pick the label\nExamples:\n- x => A"
    assert hero.skill_text(reply).startswith("**Format:** one word.") and hero.skill_text(reply).endswith("- x => A")
    assert hero.skill_text("Answer with one letter.\n- A for support") == "Answer with one letter.\n- A for support"
    assert hero.skill_text("   ") == ""
    skill = hero.skill_from_proposal("t", "Do X: ", reply, tok, 6)
    assert skill == {"id": "t", "promptPrefix": "Do X: ", "instructions": "**Format:** one word. Rules: - pick"}


def test_cross_entropy_matches_unchunked_oracle():
    model = tiny()
    head = hero.backbone_and_head(model)[1].weight.detach().float()
    pairs = [hero.Pair([5, 6, 7, 8], [9, 10, 11]), hero.Pair([3, 4], [12, 13, 14, 15, 16])]
    model.zero_grad()
    values = hero.Divergence(head, token_budget=8, chunk=2).cross_entropy(model, pairs, 1.0)
    ours = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    model.zero_grad()
    expected = torch.stack([F.cross_entropy(model(input_ids=torch.tensor([p.student + p.completion[:-1]]))
                                            .logits[0, len(p.student) - 1:].float(), torch.tensor(p.completion))
                            for p in pairs])
    expected.mean().backward()
    assert torch.allclose(torch.tensor(values), expected.detach(), atol=1e-5)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert torch.allclose(ours[name], parameter.grad, atol=1e-5, rtol=1e-4), name


def test_stage_without_anchors_audits_drift_but_never_constrains():
    model = tiny()
    head = hero.backbone_and_head(model)[1].weight.detach().float()
    divergence = hero.Divergence(head)
    probes = [hero.Pair([5, 6, 7], [8, 9, 10], group="a")]
    hero.anchor_targets(model, probes)
    records = [{"studentIds": [3, 4, 5], "teacherIds": [1, 3, 4, 5], "goldIds": [6, 7]},
               {"studentIds": [8, 9], "teacherIds": [1, 8, 9], "goldIds": [10]}]
    rows = []
    config = hero.StageConfig(updates=3, batch_size=2, checkpoint_budget=None, general_budget=1e-9)
    summary = hero.train_stage(model, records, divergence=divergence, historical=probes, general=probes,
                               config=config, log=rows.append)
    assert summary["updates"] == 3 and len(rows) == 3
    assert all(r["checkpointMultiplier"] == 0.0 for r in rows)
    assert rows[-1]["generalMultiplier"] > 0  # the budgeted anchor is enforced on the same drift
    sft = hero.train_stage(model, records, divergence=divergence, historical=[], general=[],
                           config=hero.StageConfig(updates=1, batch_size=2, objective="sft"), log=lambda row: None)
    assert sft["objective"] == "sft"


def test_window_stops_at_the_first_overflow_and_memory_keeps_the_highest_residuals():
    import run

    stream = object.__new__(run.Stream)
    stream.plan = {"seed": 42, "harness": {"windowBudgetTokens": 30, "memory": "residual",
                                           "residualBudgetTokens": 40, "generalReserveTokens": 10}}
    stream.tokens = lambda text: text.split()
    records = [{"id": str(k), "prompt": "w " * size, "answer": "a"} for k, size in enumerate((5, 20, 3, 4))]
    window = stream.window({"stage": 1, "taskId": "t"}, records)
    order = records[:]
    random.Random(43).shuffle(order)
    expected, used = [], 0
    for r in order:
        size = len(r["prompt"].split()) + 1 + 16
        if used + size > 30:
            break
        expected.append(r["id"])
        used += size
    assert [r["id"] for r in window] == expected and all(r["taskId"] == "t" for r in window)
    stream.residuals = lambda model, harness, rows: [{"a": 0.1, "b": 0.9, "c": 0.5}[r["id"]] for r in rows]
    pool = [{"id": "a", "tokens": 10}, {"id": "b", "tokens": 25}, {"id": "c", "tokens": 10}]
    assert [r["id"] for r in stream.refresh(None, hero.Harness(), pool, 2)] == ["b"]
    stream.plan["harness"]["memory"] = "uniform"
    assert sum(r["tokens"] for r in stream.refresh(None, hero.Harness(), pool, 2)) <= 30
