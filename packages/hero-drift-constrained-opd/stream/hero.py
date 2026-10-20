"""HERD stage algorithm: harness evolution and drift-constrained distillation of a privileged teacher.

Stage t receives the deployed predictor theta_{t-1} (merged weights) and the harness H_{t-1} = (K, R):

1. Evolve K. The frozen base model theta_0, shown a bounded window A_t of the task and the checkpoint's mistakes on
   it, writes the task's skill (one greedy call). The candidate registry is committed only if it lowers the mean
   residual of A_t under the frozen theta_{t-1} (verified-answer NLL with the routed skill and bare prompt in
   context, no replayed answer) and fits B_K.
2. Drift-constrained distillation. The frozen theta_{t-1}, given the routed skill and the record's verified answer
   replayed as its own previous turn, is the teacher; the student (theta_{t-1} + a fresh LoRA) sees the bare prompt
   and minimizes the full-vocabulary KL(teacher || student) at every position of the verified answer. Checkpoint
   drift (theta_{t-1} on probes from R_{t-1}) and base drift (theta_0 on the reserved general set G) are each the
   worst group's mean KL, held under a budget by a Lagrange multiplier updated by bounded dual ascent. Anchor states
   on the fixed probe sequences are computed once.
3. Merge the LoRA into the weights: theta_t.
4. Refresh R. R_t = TopBudget(R_{t-1} u A_t) by residual under theta_t executing K_t.

Only `torch`, `transformers` and `peft` are required. Sequences are lists of token ids.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
import random
import re
import time
import unicodedata
from typing import Any, Callable, Sequence

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

REPLAY = "Repeat your previous answer exactly, with no other text."
HARNESS_SYSTEM = "Apply this task skill.\n<skill>\n{instructions}\n</skill>"
PROPOSER = """You write the skill for a task: guidance that a language model reads before solving one input of the task.

Candidate examples of the task, numbered, each with its verified correct output:

{candidates}
{mistakes}
The verified output is exactly the text inside <output>; a label at the end of an input (such as "Output:" or \
"Answer:") belongs to the input. The model will see only the skill and one new input, so do not restate the task \
instructions already in the input.

Write the skill under exactly these four headers, each on its own line and in this order: "Format:", "Rules:", \
"Avoid:", "Examples:".
Under Format, one sentence on the exact format, language and length of a correct output for this task.
Under Rules, one to three lines starting with "- ", on how to decide the correct output.
Under Avoid, three to five lines starting with "- ", each a typical wrong output, then ": ", then why it is wrong, \
based on the model's mistakes above when given.
Under Examples, five to ten lines starting with "- ", each a short input, then " => ", then its correct output. Take \
them from the candidates above, keep the task's language, and keep each example under 30 words: shorten, abstract or \
merge long inputs and outputs, mark cuts with "...", and keep only what decides the output.

Start your reply with "Format:". Keep the whole skill under 350 words."""
MISTAKES = """
The model's current outputs for some of these inputs are wrong; base the Avoid list on mistakes like these:

{items}
"""


# ---------------------------------------------------------------------------
# Prompts and the bounded harness


def chat_ids(tokenizer, user: str | list[dict[str, str]], system: str | None = None) -> list[int]:
    """Prompt ids for one user turn (or a full message list) with an optional system message."""
    turns = [{"role": "user", "content": user}] if isinstance(user, str) else list(user)
    messages = ([{"role": "system", "content": system}] if system else []) + turns
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                            enable_thinking=False, return_dict=True)
    return list(encoded["input_ids"])


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split()).lower()


@dataclass(frozen=True)
class Harness:
    """hero-harness-spec-v1 registry: each prompt executes only the skill with the longest matching prefix."""
    skills: tuple[dict[str, str], ...] = ()

    def route(self, prompt: str) -> dict[str, str] | None:
        key = normalize(prompt)
        matches = [s for s in self.skills if key.startswith(normalize(s["promptPrefix"]))]
        return max(matches, key=lambda s: len(normalize(s["promptPrefix"])), default=None)

    def system(self, prompt: str) -> str | None:
        skill = self.route(prompt)
        return HARNESS_SYSTEM.format(instructions=skill["instructions"]) if skill else None

    def with_skill(self, skill: dict[str, str]) -> "Harness":
        prefix = normalize(skill["promptPrefix"])
        kept = tuple(s for s in self.skills if normalize(s["promptPrefix"]) != prefix and s["id"] != skill["id"])
        return Harness(kept + (skill,))

    def serialize(self) -> str:
        return json.dumps({"schemaVersion": "hero-harness-spec-v1", "skills": list(self.skills)},
                          ensure_ascii=False, sort_keys=True)

    def tokens(self, tokenizer) -> int:
        return len(tokenizer(self.serialize())["input_ids"])


def common_prefix(prompts: Sequence[str]) -> str:
    """Longest shared prompt prefix, cut back to a whitespace boundary when one exists."""
    first, last = min(prompts), max(prompts)
    size = next((i for i, (a, b) in enumerate(zip(first, last)) if a != b), min(len(first), len(last)))
    prefix = first[:size]
    cut = max(prefix.rfind(" "), prefix.rfind("\n"))
    return prefix[:cut + 1] if cut > 0 else prefix


def distinct_first(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Records whose answers open differently come first, so a task with several output conventions shows several;
    examples that repeat one convention bias the skill toward it."""
    distinct, rest, openings = [], [], set()
    for record in records:
        opening = " ".join(record["answer"].lower().split()[:2])
        (rest if opening in openings else distinct).append(record)
        openings.add(opening)
    return distinct + rest


def clip(tokenizer, text: str, tokens: int, tail: int = 0) -> str:
    """At most `tokens` tokens: the head, and the last `tail` of them taken from the end when the text is longer."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= tokens:
        return text
    head = tokenizer.decode(ids[:tokens - tail])
    return head + " ... " + tokenizer.decode(ids[-tail:]) if tail else head + " ..."


def proposal_prompt(candidates: Sequence[dict[str, Any]], mistakes: Sequence[tuple[dict[str, Any], str]], tokenizer,
                    *, input_tokens: int, output_tokens: int) -> str:
    """The proposer's request: numbered candidate examples with verified outputs, the model's current mistakes, and
    the required skill form. A long input keeps its head (the task definition) and its last third (the instance and
    its closing label); a long output keeps its head."""
    shown = lambda r: clip(tokenizer, r["prompt"], input_tokens, input_tokens // 3)
    items = "\n\n".join(f'<candidate n="{n}">\n<input>\n{shown(r)}\n</input>\n<output>\n'
                         f'{clip(tokenizer, r["answer"], output_tokens)}\n</output>\n</candidate>'
                         for n, r in enumerate(candidates, 1))
    wrong = "\n\n".join(f"<mistake>\n<input>\n{shown(r)}\n</input>\n<model_output>\n"
                         f"{clip(tokenizer, prediction, output_tokens)}\n</model_output>\n<output>\n"
                         f'{clip(tokenizer, r["answer"], output_tokens)}\n</output>\n</mistake>'
                         for r, prediction in mistakes)
    return PROPOSER.format(candidates=items, mistakes=MISTAKES.format(items=wrong) if mistakes else "")


def skill_text(reply: str) -> str:
    """The skill in the proposer's reply: from its Format line on when there is one (dropping any preamble), else the
    whole reply. Form is not a reason to reject a skill; only the harm check is."""
    lines = reply.strip().splitlines()
    first = next((k for k, line in enumerate(lines) if line.strip().lstrip("*#").strip().lower().startswith("format:")), 0)
    return "\n".join(lines[first:]).strip()


def skill_from_proposal(task_id: str, prompt_prefix: str, reply: str, tokenizer, max_tokens: int) -> dict[str, str]:
    """The task's skill: its routing prefix and the proposer's skill text, clipped to max_tokens."""
    ids = tokenizer(skill_text(reply), add_special_tokens=False)["input_ids"][:max_tokens]
    return {"id": task_id, "promptPrefix": prompt_prefix, "instructions": tokenizer.decode(ids).strip()}


def teacher_ids(tokenizer, prompt: str, skill: dict[str, str] | None, target: str) -> list[int]:
    """Privileged context: the routed skill (as in harness execution) and the verified target replayed as the
    model's own previous answer, then a request to repeat it. The target conditions the teacher strongly without
    any wording a student could copy: a target announced in the prompt ("The verified answer is ...") invites the
    student to learn the announcement, and a target in the system message is largely ignored."""
    system = HARNESS_SYSTEM.format(instructions=skill["instructions"]) if skill else None
    turns = [{"role": "user", "content": prompt}, {"role": "assistant", "content": target},
             {"role": "user", "content": REPLAY}]
    return chat_ids(tokenizer, turns, system)


def top_budget(records: Sequence[dict[str, Any]], residuals: dict[str, float], budget: int) -> list[dict[str, Any]]:
    """Decreasing residual, stable-id ties, one scan; a record enters only if its whole serialization fits."""
    kept, used = [], 0
    for record in sorted(records, key=lambda r: (-residuals[r["id"]], r["id"])):
        if used + record["tokens"] <= budget:
            kept.append(record)
            used += record["tokens"]
    return kept


# ---------------------------------------------------------------------------
# Model access


def backbone_and_head(model):
    """The text decoder (final-norm hidden states) and LM head. The text decoder is called directly: the
    multimodal wrapper keeps rope offsets from the last `generate` call, which do not apply to plain text."""
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    return inner.model.language_model, inner.lm_head


def eos_ids(model, tokenizer) -> list[int]:
    configured = model.generation_config.eos_token_id
    return sorted(set(([configured] if isinstance(configured, int) else list(configured or []))
                      + [tokenizer.eos_token_id]))


def _token_batches(lengths: Sequence[int], budget: int, max_rows: int) -> list[list[int]]:
    """Index groups, longest first, with rows x longest row within the token budget."""
    groups, current, width = [], [], 0
    for k in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        if current and (max(width, lengths[k]) * (len(current) + 1) > budget or len(current) == max_rows):
            groups.append(current)
            current, width = [], 0
        current.append(k)
        width = max(width, lengths[k])
    return groups + ([current] if current else [])


@torch.no_grad()
def generate(model, prompts: Sequence[list[int]], *, max_new_tokens: int, eos: Sequence[int], pad_id: int,
             token_budget: int = 65536, max_rows: int = 64) -> list[dict[str, Any]]:
    """Left-padded batched greedy decoding. Returns completion ids (through the first EOS) and the stop reason."""
    from transformers import GenerationConfig

    device = backbone_and_head(model)[1].weight.device
    config = GenerationConfig(max_new_tokens=max_new_tokens, eos_token_id=list(eos), pad_token_id=pad_id,
                              use_cache=True, do_sample=False)
    was_training = model.training
    model.eval()
    results: list[dict[str, Any]] = [None] * len(prompts)  # type: ignore[list-item]
    for group in _token_batches([len(p) + max_new_tokens for p in prompts], token_budget, max_rows):
        width = max(len(prompts[k]) for k in group)
        ids = torch.full((len(group), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(group), width), dtype=torch.long)
        for row, k in enumerate(group):
            ids[row, width - len(prompts[k]):] = torch.tensor(prompts[k])
            mask[row, width - len(prompts[k]):] = 1
        output = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device), generation_config=config)
        for row, k in enumerate(group):
            tokens = output[row, width:].tolist()
            stop = next((i for i, t in enumerate(tokens) if t in eos), None)
            results[k] = {"tokens": tokens[:stop + 1] if stop is not None else tokens,
                          "stopReason": "eos" if stop is not None else "length"}
    if was_training:
        model.train()
    return results


def completion_hidden(model, rows: Sequence[tuple[list[int], list[int]]]) -> torch.Tensor:
    """Final hidden states at the positions that predict each completion token, concatenated over rows."""
    backbone, head = backbone_and_head(model)
    device = head.weight.device
    sequences = [prompt + completion[:-1] for prompt, completion in rows]
    width = max(len(s) for s in sequences)
    ids = torch.zeros((len(rows), width), dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    index_rows, index_cols = [], []
    for row, ((prompt, completion), sequence) in enumerate(zip(rows, sequences)):
        ids[row, :len(sequence)] = torch.tensor(sequence)
        mask[row, :len(sequence)] = 1
        index_rows += [row] * len(completion)
        index_cols += range(len(prompt) - 1, len(prompt) - 1 + len(completion))
    hidden = backbone(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False).last_hidden_state
    return hidden[torch.tensor(index_rows, device=device), torch.tensor(index_cols, device=device)]


def _token_kl(student: torch.Tensor, target: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
    """Forward KL(target || student) per token over the full vocabulary, in FP32."""
    with torch.autocast(device_type=student.device.type, enabled=False):
        log_s = F.log_softmax(student.float() @ head.t(), dim=-1)
        with torch.no_grad():
            log_t = F.log_softmax(target.float() @ head.t(), dim=-1)
        return (log_t.exp() * (log_t - log_s)).sum(-1)


def _token_ce(student: torch.Tensor, labels: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of the verified tokens per position, in FP32 (the SFT-objective ablation)."""
    with torch.autocast(device_type=student.device.type, enabled=False):
        return F.cross_entropy(student.float() @ head.t(), labels, reduction="none")


@dataclass
class Pair:
    """Student input, completion and the anchor/teacher view of the same completion."""
    student: list[int]
    completion: list[int]
    teacher: list[int] | None = None  # teacher prompt; its hidden states are computed with the adapter disabled
    target: torch.Tensor | None = None  # precomputed anchor hidden states [len(completion), hidden]
    group: str = ""  # probe group (historical task or general source); drift is the worst group's mean


class Divergence:
    """Mean-over-sequences of a per-token loss, accumulated into .grad in token-budget microbatches."""

    def __init__(self, head32: torch.Tensor, *, token_budget: int = 16384, chunk: int = 512) -> None:
        self.head32, self.budget, self.chunk = head32, token_budget, chunk

    @torch.no_grad()
    def attach_teacher(self, model, pairs: Sequence[Pair]) -> None:
        with frozen(model):
            for group in _token_batches([len(p.teacher) + len(p.completion) for p in pairs], self.budget * 2, 64):
                hidden = completion_hidden(model, [(pairs[k].teacher, pairs[k].completion) for k in group])
                for k, part in zip(group, hidden.split([len(pairs[k].completion) for k in group])):
                    pairs[k].target = part

    def __call__(self, model, pairs: Sequence[Pair], weight: float) -> list[float]:
        """KL(target || student): backpropagate weight * mean_i KL_i when weight > 0; always return each pair's KL."""
        return self._accumulate(model, pairs, weight, _token_kl,
                                lambda group, device: torch.cat([pairs[k].target.to(device) for k in group]))

    def cross_entropy(self, model, pairs: Sequence[Pair], weight: float) -> list[float]:
        """The same accumulation for the verified completion's cross-entropy under the student."""
        return self._accumulate(model, pairs, weight, _token_ce, lambda group, device: torch.tensor(
            [t for k in group for t in pairs[k].completion], device=device))

    def _accumulate(self, model, pairs: Sequence[Pair], weight: float, per_token_loss, targets) -> list[float]:
        values: list[float] = [0.0] * len(pairs)
        grad = weight > 0 and torch.is_grad_enabled()
        for group in _token_batches([len(p.student) + len(p.completion) for p in pairs], self.budget, 64):
            with torch.set_grad_enabled(grad):
                hidden = completion_hidden(model, [(pairs[k].student, pairs[k].completion) for k in group])
                target = targets(group, hidden.device)
                scale = torch.cat([torch.full((len(pairs[k].completion),), 1.0 / len(pairs[k].completion))
                                   for k in group]).to(hidden.device)
                per_token = torch.cat([
                    checkpoint(per_token_loss, hidden[s:s + self.chunk], target[s:s + self.chunk], self.head32,
                               use_reentrant=False) if grad else
                    per_token_loss(hidden[s:s + self.chunk], target[s:s + self.chunk], self.head32)
                    for s in range(0, hidden.shape[0], self.chunk)])
                if grad:
                    (per_token * scale).sum().mul(weight / len(pairs)).backward()
            for k, part in zip(group, (per_token.detach() * scale).split([len(pairs[k].completion) for k in group])):
                values[k] = float(part.sum())
            del hidden, target, per_token
        return values


def frozen(model):
    """The frozen checkpoint: the model with its LoRA disabled (or the plain model before any LoRA)."""
    return model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()


def initialize_adapter(model, config, *, seed: int, stage: int):
    """Stage-local RNG makes initialization independent of prior stages or a resume."""
    from peft import get_peft_model

    torch.manual_seed(seed + stage)
    return get_peft_model(model, config)


@torch.no_grad()
def anchor_targets(model, probes: Sequence[Pair], budget: int = 32768) -> None:
    """Fill each fixed probe's target with the frozen model's hidden states on its own completion."""
    with frozen(model):
        for group in _token_batches([len(p.student) + len(p.completion) for p in probes], budget, 64):
            hidden = completion_hidden(model, [(probes[k].student, probes[k].completion) for k in group])
            for k, part in zip(group, hidden.split([len(probes[k].completion) for k in group])):
                probes[k].target = part.cpu()


@torch.no_grad()
def target_nll(model, rows: Sequence[tuple[list[int], list[int]]], head32: torch.Tensor,
               budget: int = 32768) -> list[float]:
    """Mean target NLL per (prompt, target) row under the frozen model: the residual when the prompt holds
    only the routed skill and the bare input."""
    values: list[float] = [0.0] * len(rows)
    with frozen(model):
        for group in _token_batches([len(p) + len(c) for p, c in rows], budget, 64):
            hidden = completion_hidden(model, [rows[k] for k in group])
            labels = torch.tensor([t for k in group for t in rows[k][1]], device=hidden.device)
            losses = torch.cat([F.cross_entropy(hidden[s:s + 512].float() @ head32.t(), labels[s:s + 512],
                                                reduction="none") for s in range(0, len(labels), 512)])
            for k, part in zip(group, losses.split([len(rows[k][1]) for k in group])):
                values[k] = float(part.mean())
    return values


# ---------------------------------------------------------------------------
# Drift-constrained distillation for one stage


@dataclass(frozen=True)
class StageConfig:
    updates: int = 32
    batch_size: int = 32
    learning_rate: float = 1e-4
    gradient_clip: float = 1.0
    checkpoint_budget: float | None = 0.05  # None: drift measured and audited, never constrained (ablation)
    general_budget: float | None = 0.05
    dual_rate: float = 0.5
    objective: str = "distill"  # "sft": cross-entropy on the verified answer (ablation)
    seed: int = 42


def batch_schedule(size: int, *, updates: int, batch_size: int, seed: int) -> list[list[int]]:
    """Seeded shuffle, reshuffled at each epoch boundary."""
    order, rng, schedule, cursor = list(range(size)), random.Random(seed), [], 0
    rng.shuffle(order)
    for _ in range(updates):
        batch = []
        for _ in range(batch_size):
            if cursor == size:
                rng.shuffle(order)
                cursor = 0
            batch.append(order[cursor])
            cursor += 1
        schedule.append(batch)
    return schedule


def train_stage(model, records: Sequence[dict[str, Any]], *, divergence: Divergence,
                historical: Sequence[Pair], general: Sequence[Pair], config: StageConfig,
                log: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Distill the privileged teacher into the LoRA of `model` on the verified answers, under the drift budgets.

    records: {"studentIds", "teacherIds", "goldIds"}. The loss is the full-vocabulary KL(teacher || student) at every
    position of the verified answer. Training on the verified answer's own prefix, rather than on the student's
    samples, closes the student's degenerate paths (such as prompt echo), on which teacher and student agree and
    the gradient vanishes. Each update measures both drifts without gradient, adds every active constraint's
    worst-group divergence with weight lambda/epsilon, steps on the adapter, and moves the multipliers by bounded
    dual ascent from the same measurement. The learning rate decays linearly to zero; the final iterate is kept."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=0.0)
    decay = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 - step / config.updates)
    budgets = {"checkpoint": config.checkpoint_budget, "general": config.general_budget}
    probes = {"checkpoint": list(historical), "general": list(general)}
    multipliers = {"checkpoint": 0.0, "general": 0.0}
    schedule = batch_schedule(len(records), updates=config.updates, batch_size=config.batch_size, seed=config.seed)
    model.train()
    history = []
    for update, indices in enumerate(schedule, start=1):
        started = time.perf_counter()
        pairs = [Pair(records[i]["studentIds"], records[i]["goldIds"], teacher=records[i]["teacherIds"])
                 for i in indices]
        optimizer.zero_grad(set_to_none=True)
        if config.objective == "sft":
            loss = divergence.cross_entropy(model, pairs, 1.0)
        else:
            divergence.attach_teacher(model, pairs)
            loss = divergence(model, pairs, 1.0)
        drift = {}
        for name in ("checkpoint", "general"):
            drift[name], worst = group_drift(divergence, model, probes[name])
            if budgets[name] is not None and multipliers[name] > 0 and worst:
                divergence(model, worst, multipliers[name] / budgets[name])
        norm = float(torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip))
        optimizer.step()
        decay.step()
        for name in multipliers:
            if budgets[name] is not None and probes[name]:
                violation = min(1.0, drift[name] / budgets[name] - 1)  # bounded dual step
                multipliers[name] = max(0.0, multipliers[name] + config.dual_rate * violation)
        row = {"update": update, "loss": sum(loss) / len(loss), "checkpointDrift": drift["checkpoint"],
               "generalDrift": drift["general"], "checkpointMultiplier": multipliers["checkpoint"],
               "generalMultiplier": multipliers["general"], "gradNorm": norm,
               "targetTokens": sum(len(p.completion) for p in pairs), "seconds": time.perf_counter() - started}
        history.append(row)
        log(row)
    tail = history[-max(1, len(history) // 4):]
    return {"updates": len(history), "records": config.updates * config.batch_size, "objective": config.objective,
            "lossFirst": history[0]["loss"], "lossLastQuarter": sum(r["loss"] for r in tail) / len(tail),
            "seconds": sum(r["seconds"] for r in history)}


@torch.no_grad()
def group_drift(divergence: Divergence, model, probes: Sequence[Pair]) -> tuple[float, list[Pair]]:
    """Worst probe group's mean KL and that group's probes. A mean over all probes would let one group (one task,
    or one general source) drift far while the overall mean stays within budget."""
    if not probes:
        return 0.0, []
    groups: dict[str, list[int]] = {}
    for k, probe in enumerate(probes):
        groups.setdefault(probe.group, []).append(k)
    values = divergence(model, probes, 0.0)
    means = {g: sum(values[k] for k in ks) / len(ks) for g, ks in groups.items()}
    worst = max(means, key=means.get)
    return means[worst], [probes[k] for k in groups[worst]]


def measure_drift(model, divergence: Divergence, probes: Sequence[Pair]) -> float | None:
    return group_drift(divergence, model, probes)[0] if probes else None
