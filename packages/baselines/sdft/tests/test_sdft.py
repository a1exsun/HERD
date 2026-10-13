import math

import pytest
import torch

from sdft import (
    TEACHER_TEMPLATE, EmaTeacher, learning_rate_at, prompt_ids, reference_forward_kl, sequence_forward_kl,
    sequence_loss, teacher_prompt, update_batches,
)


class TinyDecoder(torch.nn.Module):
    def __init__(self, vocab=11, hidden=6):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.mix = torch.nn.Linear(hidden, hidden)

    def forward(self, input_ids, position_ids, use_cache):
        states = self.embed(input_ids)
        # Causal prefix mean, so each position depends on its context like a decoder.
        prefix = states.cumsum(dim=1) / torch.arange(1, states.shape[1] + 1)[None, :, None]
        return (torch.tanh(self.mix(prefix)),)


class TinyBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyDecoder()
        self.lm_head = torch.nn.Linear(6, 11, bias=False)
        self.lm_head.weight.requires_grad_(False)


class TinyPeft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = TinyBase()

    def get_base_model(self):
        return self.base


def full_logits(model, prompt, completion):
    ids = torch.tensor([prompt + completion[:-1]])
    hidden = model.base.model(ids, torch.arange(ids.shape[1])[None], False)[0][0, len(prompt) - 1:]
    return hidden @ model.base.lm_head.weight.T


def test_chunked_forward_kl_matches_unchunked_reference_and_its_gradient():
    torch.manual_seed(0)
    model = TinyPeft()
    student_prompt, teacher_prompt_ids, completion = [1, 2], [3, 4, 5, 6], [7, 8, 9, 2, 10]
    with torch.no_grad():
        teacher_hidden = model.base.model(torch.tensor([teacher_prompt_ids + completion[:-1]]),
                                          torch.arange(len(teacher_prompt_ids) + 4)[None], False)[0][0, 3:]
    per_token, logps = sequence_forward_kl(model, student_prompt, teacher_hidden, completion, chunk_tokens=2)
    per_token.mean().backward()
    chunked_grad = model.base.model.mix.weight.grad.clone()
    model.zero_grad()
    student_logits = full_logits(model, student_prompt, completion)
    reference = reference_forward_kl(student_logits, full_logits(model, teacher_prompt_ids, completion))
    reference.mean().backward()
    assert torch.allclose(per_token, reference, rtol=1e-5, atol=1e-7)
    assert torch.allclose(chunked_grad, model.base.model.mix.weight.grad, atol=1e-6)
    expected = torch.log_softmax(student_logits, -1).gather(-1, torch.tensor(completion)[:, None]).squeeze(-1)
    assert torch.allclose(logps, expected.detach(), atol=1e-6) and not logps.requires_grad


def test_forward_kl_direction_is_teacher_to_student():
    student = torch.tensor([[2.0, 0.0, -1.0]])
    teacher = torch.tensor([[0.0, 1.0, 0.0]])
    p, q = torch.softmax(teacher, -1), torch.softmax(student, -1)
    expected = (p * (p.log() - q.log())).sum()
    assert reference_forward_kl(student, teacher).item() == pytest.approx(expected.item(), rel=1e-6)


def test_loss_masks_the_first_three_tokens_and_takes_the_token_mean():
    per_token = torch.tensor([10.0, 20.0, 30.0, 1.0, 3.0])
    loss, weight = sequence_loss(per_token, torch.zeros(5), None, skip=3, cap=2.0)
    assert loss.item() == pytest.approx(2.0) and weight is None
    short, _ = sequence_loss(torch.tensor([4.0, 5.0], requires_grad=True), torch.zeros(2), None, skip=3, cap=2.0)
    assert short.item() == 0.0  # a completion of at most three tokens carries no loss, as upstream
    short.backward()


def test_importance_weight_is_the_capped_ratio_mean_over_loss_tokens():
    per_token = torch.tensor([9.0, 9.0, 9.0, 2.0, 4.0])
    student = torch.tensor([0.0, 0.0, 0.0, math.log(0.5), math.log(8.0)])
    sampler = torch.zeros(5)
    loss, weight = sequence_loss(per_token, student, sampler, skip=3, cap=2.0)
    assert weight == pytest.approx((0.5 + 2.0) / 2)  # ratios 0.5 and 8 -> capped at 2
    assert loss.item() == pytest.approx(1.25 * 3.0)


class TinyAdapted(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 3, bias=False)
        self.norm = torch.nn.Parameter(torch.ones(4))


def peft_student(seed=0):
    import copy

    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    base = TinyAdapted()
    teacher_copy = copy.deepcopy(base)
    student = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["proj"]))
    with torch.no_grad():
        for name, parameter in student.named_parameters():
            if "lora_B" in name:
                parameter.normal_()
    return student, teacher_copy


def test_teacher_tracks_the_students_effective_weight_at_every_sync():
    import copy

    student, teacher_copy = peft_student()
    merged = copy.deepcopy(student).merge_and_unload()
    teacher = EmaTeacher(teacher_copy, student, alpha=0.25)
    assert torch.allclose(teacher.student_weight("proj.weight"), merged.proj.weight, atol=1e-6)
    assert torch.equal(teacher.student_weight("norm"), student.get_base_model().norm.detach())
    before = teacher.model.proj.weight.detach().clone()
    teacher.update()
    assert torch.allclose(teacher.model.proj.weight, 0.75 * before + 0.25 * merged.proj.weight, atol=1e-6)
    assert torch.allclose(teacher.model.norm, torch.ones(4))  # frozen weights: ref stays W0
    assert not any(p.requires_grad for p in teacher.model.parameters())
    state = teacher.state_dict()
    _, other_copy = peft_student(seed=1)
    restored = EmaTeacher(other_copy, student, alpha=0.25)
    restored.load_state_dict(state)
    assert all(torch.equal(a, b) for a, b in zip(restored.model.parameters(), teacher.model.parameters()))
    with pytest.raises(ValueError, match="does not match"):
        restored.load_state_dict({"norm": state["norm"]})


def test_teacher_ema_runs_in_the_parameter_dtype():
    student, teacher_copy = peft_student()
    student.get_base_model().proj.base_layer.to(torch.bfloat16)  # bf16 base weights, fp32 LoRA factors
    teacher = EmaTeacher(teacher_copy.to(torch.bfloat16), student, alpha=0.01)
    assert teacher.student_weight("proj.weight").dtype == torch.bfloat16
    teacher.update()
    assert teacher.model.proj.weight.dtype == torch.bfloat16


def test_teacher_prompt_template_is_verbatim():
    record = {"id": "x", "prompt": "Q?", "answer": "", "references": ["gold", "other"]}
    assert teacher_prompt(record) == ("\nQ?\n\nThis is an example for a response to the question:\ngold\n\n"
                                      "Now answer with a response of your own, including the thinking process.\n")
    assert TEACHER_TEMPLATE.substitute(orig_content="Q?", output_text="gold") == teacher_prompt(record)


def test_learning_rate_warms_up_then_decays_to_zero():
    rates = [learning_rate_at(1.0, update, 32, 0.1) for update in range(32)]
    assert rates[:4] == [0.0, 0.25, 0.5, 0.75]
    assert rates[4] == pytest.approx(1.0)
    assert rates[-1] == pytest.approx(0.5 * (1 + math.cos(math.pi * 27 / 28)))
    assert all(a >= b for a, b in zip(rates[4:], rates[5:]))


def test_update_batches_follow_dataset_shuffle_then_repeat_sampler():
    import numpy as np

    size, batch, seed = 10, 4, 42
    shuffled = np.random.default_rng(seed).permutation(size).tolist()
    generator = torch.Generator()
    generator.manual_seed(seed)
    expected = []
    for _ in range(2):  # two full batches per epoch; the remaining two records are dropped
        order = torch.randperm(size, generator=generator).tolist()
        expected += [[shuffled[i] for i in order[0:4]], [shuffled[i] for i in order[4:8]]]
    assert update_batches(size, updates=3, batch_size=batch, seed=seed) == expected[:3]
    with pytest.raises(ValueError):
        update_batches(3, updates=1, batch_size=4, seed=seed)


class LengthTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": list(range(len(messages[-1]["content"])))}


class TinyTokenizer:
    """One id per character, inside the tiny model's vocabulary."""
    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": [ord(c) % 11 for c in messages[-1]["content"]]}


def test_prompts_beyond_the_ceiling_are_left_truncated():
    assert prompt_ids(LengthTokenizer(), "abc", 5) == ([0, 1, 2], False)
    assert prompt_ids(LengthTokenizer(), "abcdefg", 5) == ([2, 3, 4, 5, 6], True)


def test_train_stage_runs_end_to_end_with_server_rollouts():
    from sdft import Rollout, SdftConfig, train_stage

    import copy

    torch.manual_seed(0)
    model = TinyPeft()
    model.base.model.embed.weight.requires_grad_(False)
    model.base.model.mix.bias.requires_grad_(False)
    teacher = EmaTeacher(copy.deepcopy(model.base), model, alpha=0.01)
    records = [{"id": str(i), "prompt": "p" * (i + 1), "answer": "a"} for i in range(6)]
    calls = []

    def sampler(model, tokenizer, encoded, *, config, eos):
        calls.append([len(ids) for ids in encoded])
        return [Rollout("x", (7, 8, 9, 2, 10), "length", (-1.0,) * 5) for _ in encoded]

    rows = []
    summary = train_stage(model, TinyTokenizer(), stage=1, task_id="t", records=records, updates=3, batch_size=2,
                          learning_rate=1e-2, gradient_clip=1.0, seed=42, eos=[10], config=SdftConfig(),
                          teacher=teacher, log=rows.append, sampler=sampler)
    assert len(calls) == 3 and summary.updates == 3 and summary.truncated_prompts == []
    samples = [row for row in rows if row["kind"] == "sample"]
    assert len(samples) == 6 and all(row["lossTokens"] == 2 and row["importanceWeight"] > 0 for row in samples)
    lag = teacher.model.model.mix.weight - model.base.model.mix.weight
    assert lag.abs().max() > 0  # the EMA teacher lags the student


def test_endpoint_returns_sampler_logprobs_and_rejects_missing_ones(monkeypatch, tmp_path):
    import endpoint

    monkeypatch.setattr(endpoint, "_get", lambda host, port, path: {"version": "x"} if path == "/version"
                        else {"data": [{"id": "base"}]})
    replies = {"logprobs": {"token_logprobs": [-0.5, -0.25, -1.0]}, "token_ids": [5, 6, 3]}
    bodies = []

    def post(host, port, path, body):
        bodies.append((path, body))
        return {"choices": [dict(replies)]} if path == "/v1/completions" else "ok"

    monkeypatch.setattr(endpoint, "_post", post)
    monkeypatch.setattr(endpoint, "write_stacked_adapter", lambda parts, config, directory: None)

    class Tokenizer:
        def decode(self, tokens, skip_special_tokens):
            return " ".join(map(str, tokens))

    sampler = endpoint.EndpointSampler("http://server:8000", [], tmp_path, "t", workers=1)
    monkeypatch.setattr(sampler, "_live", lambda model, directory: ({}, {"r": 32, "lora_alpha": 64}))
    from sdft import SdftConfig

    rollout = sampler(None, Tokenizer(), [[1, 2]], config=SdftConfig(), eos=[6])[0]
    assert rollout.token_ids == (5, 6) and rollout.stop_reason == "eos"
    assert rollout.sampling_logprobs == (-0.5, -0.25)
    request = next(body for path, body in bodies if path == "/v1/completions")
    assert request["logprobs"] == 0 and "seed" not in request and request["top_k"] == -1
    assert sampler.server == {"version": {"version": "x"}, "models": ["base"]}
    replies["logprobs"] = {"token_logprobs": [-0.5]}
    with pytest.raises(RuntimeError, match="no log-probability"):
        sampler(None, Tokenizer(), [[1, 2]], config=SdftConfig(), eos=[6])
