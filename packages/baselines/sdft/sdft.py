"""Self-Distillation Fine-Tuning (SDFT, arXiv:2601.19897), self-contained.

For a demonstration (x, c), the student samples one completion y ~ pi_theta(.|x)
from the bare prompt. The teacher is the same model conditioned on the demonstration, pi_phi(.|x, c), whose
weights are an exponential moving average of the student's, phi <- (1 - alpha) phi + alpha theta after every
optimizer step, held as a separate copy of the model. The loss is the full-vocabulary forward KL(teacher || student) per completion
token, computed in the model dtype, with the first three completion tokens masked; with rollouts from a vLLM
server each sequence is weighted by the mean of its capped per-token importance ratio; then the token mean over
the completion and the mean over the batch.

Under the shared LoRA budget the student's weight of an adapted Linear layer is its effective weight
W0 + scaling * B A (PEFT's merge delta); every other student weight is the frozen W0. The teacher's EMA runs over
all of its parameters against those weights, in the model dtype, at every sync.

Only torch/transformers/peft/numpy are imported, lazily, so the pure logic is testable without a model.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
from string import Template
import time
from typing import Any, Callable, Iterator, Mapping, Sequence


# Teacher prompt template, verbatim.
TEACHER_TEMPLATE = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")


@dataclass(frozen=True)
class SdftConfig:
    # Paper run configuration.
    ema_alpha: float = 0.01                 # ref_model_mixup_alpha, synced every optimizer step
    rollout_temperature: float = 1.0        # sampling temperature; logits are divided by it in the loss
    rollout_max_new_tokens: int = 1024      # max_completion_length
    warmup_ratio: float = 0.1               # cosine schedule with linear warmup
    num_loss_tokens_to_skip: int = 3
    importance_sampling_cap: float = 2.0    # vllm_importance_sampling_cap, applied to vLLM rollouts
    # Project protocol: the input ceiling replaces the method's 1024 prompt cap (still left-truncating).
    max_prompt_length: int = 16384
    # Execution only; no effect on the objective.
    generation_batch_size: int = 32
    generation_token_budget: int = 450_000
    generation_prefill_budget: int = 65_536
    kl_chunk_tokens: int = 1024


EXECUTION_ONLY_FIELDS = frozenset({"generation_batch_size", "generation_token_budget",
                                   "generation_prefill_budget", "kl_chunk_tokens"})
CHECKPOINT_ENCODING = "sdft-ema-xor-gzip-v1"


def demonstration(record: Mapping[str, Any]) -> str:
    """The expert demonstration c: the record's first reference answer."""
    answer = record.get("answer")
    if isinstance(answer, str) and answer:
        return answer
    references = [value for value in record.get("references", []) if isinstance(value, str) and value]
    if references:
        return references[0]
    raise ValueError(f"Record {record.get('id')} has no demonstration.")


def teacher_prompt(record: Mapping[str, Any]) -> str:
    return TEACHER_TEMPLATE.substitute(orig_content=str(record["prompt"]), output_text=demonstration(record))


def learning_rate_at(peak: float, update: int, updates: int, warmup_ratio: float) -> float:
    """Linear warmup over ceil(ratio * updates) steps from 0, then cosine decay to 0 (the Trainer schedule)."""
    warmup = math.ceil(warmup_ratio * updates)
    if update < warmup:
        return peak * update / max(1, warmup)
    progress = (update - warmup) / max(1, updates - warmup)
    return peak * max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def update_batches(size: int, *, updates: int, batch_size: int, seed: int) -> list[list[int]]:
    """Record indices per update in the method's order: `Dataset.shuffle(seed)` (numpy `default_rng(seed)`
    permutation), then a repeat sampler that draws one `torch.randperm` per epoch from a generator
    seeded with `seed` and drops the incomplete last batch of every epoch."""
    import numpy as np
    import torch

    if size < batch_size:
        raise ValueError("A stage needs at least one full batch of records.")
    shuffled = np.random.default_rng(seed).permutation(size).tolist()
    generator = torch.Generator()
    generator.manual_seed(seed)
    schedule: list[list[int]] = []
    while len(schedule) < updates:
        order = torch.randperm(size, generator=generator).tolist()
        for start in range(0, size - batch_size + 1, batch_size):
            if len(schedule) == updates:
                break
            schedule.append([shuffled[index] for index in order[start:start + batch_size]])
    return schedule


# ---------------------------------------------------------------------------
# EMA teacher: a separate copy of the model


class EmaTeacher:
    """A separate model whose every parameter follows
    ref <- (1 - alpha) ref + alpha theta after each optimizer step, in the parameters' dtype.

    `model` is a copy of the student's weights at the start of the stage.
    `student` is the PEFT student; theta of an adapted Linear weight is its
    effective weight W0 + scaling * B A, and of every other parameter the frozen W0."""

    def __init__(self, model, student, alpha: float):
        if not 0 < alpha <= 1:
            raise ValueError("EMA alpha must be in (0, 1].")
        self.model = model.eval().requires_grad_(False)
        self.student = student.get_base_model() if hasattr(student, "get_base_model") else student
        self.alpha = alpha

    def student_weight(self, name: str):
        """The student's current value of the teacher parameter `name`, in that parameter's dtype."""
        path, _, attribute = name.rpartition(".")
        module = self.student.get_submodule(path)
        base = getattr(module, "base_layer", None)
        if base is None:
            return getattr(module, attribute).detach()
        value = getattr(base, attribute).detach()
        adapters = [adapter for adapter in module.active_adapters if adapter in module.lora_A]
        if attribute != "weight" or module.disable_adapters or not adapters:
            return value
        delta = sum(module.get_delta_weight(adapter).detach().float() for adapter in adapters)
        return (value.float() + delta).to(value.dtype)

    def update(self) -> None:
        import torch

        with torch.no_grad():
            for name, reference in self.model.named_parameters():
                reference.data.mul_(1.0 - self.alpha).add_(self.student_weight(name), alpha=self.alpha)

    def state_dict(self) -> dict[str, Any]:
        return {name: parameter.detach().cpu().contiguous() for name, parameter in self.model.named_parameters()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        import torch

        names = [name for name, _ in self.model.named_parameters()]
        if sorted(state) != sorted(names):
            raise ValueError("EMA teacher state does not match the teacher's parameters.")
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                parameter.copy_(state[name].to(parameter.device, dtype=parameter.dtype))

    def _base_bytes(self, name: str):
        """Frozen stage-start bytes, excluding the current adapter's delta."""
        import torch

        path, _, attribute = name.rpartition(".")
        module = self.student.get_submodule(path)
        parameter = getattr(getattr(module, "base_layer", module), attribute)
        return parameter.detach().reshape(-1).cpu().contiguous().view(torch.uint8)

    def save_checkpoint(self, path) -> None:
        """Lossless byte deltas against the frozen base; never round or omit EMA parameters."""
        import torch

        with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as stream:
            stream.write((CHECKPOINT_ENCODING + "\n").encode())
            for name, parameter in self.model.named_parameters():
                base = self._base_bytes(name)
                value = parameter.detach().reshape(-1).cpu().contiguous().view(torch.uint8)
                if base.shape != value.shape:
                    raise ValueError(f"EMA base shape mismatch: {name}")
                header = {"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype),
                          "baseSha256": hashlib.sha256(base.numpy()).hexdigest(), "bytes": value.numel()}
                stream.write((json.dumps(header, sort_keys=True) + "\n").encode())
                delta = torch.bitwise_xor(value, base).numpy()
                stream.write(memoryview(delta))

    def load_checkpoint(self, path) -> None:
        import torch

        with gzip.open(path, "rb") as stream, torch.no_grad():
            if stream.readline() != (CHECKPOINT_ENCODING + "\n").encode():
                raise ValueError("Unsupported EMA checkpoint encoding.")
            for name, parameter in self.model.named_parameters():
                header = json.loads(stream.readline())
                base = self._base_bytes(name)
                if (header["name"] != name or header["shape"] != list(parameter.shape)
                        or header["dtype"] != str(parameter.dtype) or header["bytes"] != base.numel()
                        or header["baseSha256"] != hashlib.sha256(base.numpy()).hexdigest()):
                    raise ValueError(f"EMA checkpoint base does not match: {name}")
                data = bytearray(stream.read(base.numel()))
                if len(data) != base.numel():
                    raise ValueError(f"Truncated EMA checkpoint: {name}")
                delta = torch.frombuffer(data, dtype=torch.uint8)
                value = torch.bitwise_xor(delta, base).view(parameter.dtype).reshape(parameter.shape)
                parameter.copy_(value.to(parameter.device))
            if stream.read(1):
                raise ValueError("Unexpected EMA checkpoint parameters.")


# ---------------------------------------------------------------------------
# Model-side pieces (torch imported lazily)


def chat_ids(tokenizer, prompt: str) -> list[int]:
    return list(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                              add_generation_prompt=True, enable_thinking=False,
                                              return_dict=True)["input_ids"])


def prompt_ids(tokenizer, prompt: str, max_length: int) -> tuple[list[int], bool]:
    """Chat-template ids, left-truncated to `max_length`; and whether it truncated."""
    ids = chat_ids(tokenizer, prompt)
    return (ids[-max_length:], True) if len(ids) > max_length else (ids, False)


def lora_targets(model) -> list[str]:
    """Text-backbone Linear modules; vision, MTP and the LM head stay frozen."""
    import torch

    excluded = ("visual", "vision", "mtp", "lm_head")
    targets = sorted(name for name, module in model.named_modules()
                     if isinstance(module, torch.nn.Linear) and not any(part in name.lower() for part in excluded))
    if not targets:
        raise ValueError("No text-backbone Linear modules found.")
    return targets


@dataclass(frozen=True)
class Rollout:
    text: str
    token_ids: tuple[int, ...]
    stop_reason: str
    # Per-token log-probabilities reported by an inference server; None for in-process generation.
    sampling_logprobs: tuple[float, ...] | None = None


def generation_chunks(lengths: Sequence[int], *, max_new_tokens: int, batch_size: int,
                      token_budget: int, prefill_budget: int) -> list[list[int]]:
    """Group prompt indices, shortest first, so each padded chunk fits the KV and prefill budgets."""
    order = sorted(range(len(lengths)), key=lambda index: lengths[index])
    chunks: list[list[int]] = []
    for index in order:
        chunk = chunks[-1] if chunks else None
        if chunk is not None and len(chunk) < batch_size and \
                (len(chunk) + 1) * (lengths[index] + max_new_tokens) <= token_budget and \
                (len(chunk) + 1) * lengths[index] <= prefill_budget:
            chunk.append(index)
        else:
            chunks.append([index])
    return chunks


def sample_rollouts(model, tokenizer, encoded: Sequence[Sequence[int]], *, config: SdftConfig,
                    eos: Sequence[int]) -> list[Rollout]:
    """On-policy completions from the current student (in-process sampling): left-padded batches,
    temperature 1, top-p 1, no top-k, from the global RNG seeded at the start of the stage."""
    import torch

    device = next(model.parameters()).device
    results: list[Rollout | None] = [None] * len(encoded)
    pad = tokenizer.pad_token_id
    chunks = generation_chunks([len(ids) for ids in encoded], max_new_tokens=config.rollout_max_new_tokens,
                               batch_size=config.generation_batch_size, token_budget=config.generation_token_budget,
                               prefill_budget=config.generation_prefill_budget)
    for chunk in chunks:
        width = max(len(encoded[index]) for index in chunk)
        ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for row, index in enumerate(chunk):
            tokens = list(encoded[index])
            ids[row, width - len(tokens):] = torch.tensor(tokens)
            mask[row, width - len(tokens):] = 1
        with torch.inference_mode():
            output = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device), do_sample=True,
                                    temperature=config.rollout_temperature, top_p=1.0, top_k=0,
                                    repetition_penalty=1.0, max_new_tokens=config.rollout_max_new_tokens,
                                    eos_token_id=list(eos), pad_token_id=pad)
        for row, index in enumerate(chunk):
            tokens = output[row, width:].tolist()
            stop = next((position for position, token in enumerate(tokens) if token in eos), None)
            tokens = tokens[:stop + 1] if stop is not None else tokens
            results[index] = Rollout(tokenizer.decode(tokens, skip_special_tokens=True), tuple(tokens),
                                     "eos" if stop is not None else "length")
    return [result for result in results if result is not None]


def base_model(model):
    """The underlying transformers model of a PEFT model, or the model itself."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def completion_hidden(model, prompt_ids: Sequence[int], completion_ids: Sequence[int]):
    """Final-norm hidden states at the positions that predict each completion token."""
    import torch

    decoder = base_model(model).model
    device = next(model.parameters()).device
    ids = torch.tensor([list(prompt_ids) + list(completion_ids[:-1])], dtype=torch.long, device=device)
    # Text-only mRoPE positions are plain arange; passing them avoids rope deltas cached by generate().
    positions = torch.arange(ids.shape[1], device=device)[None]
    hidden = decoder(input_ids=ids, position_ids=positions, use_cache=False)[0]
    return hidden[0, len(prompt_ids) - 1:]


def _chunk_forward_kl(student_hidden, teacher_hidden, weight, teacher_weight, tokens, temperature):
    """Per-token loss on one chunk, in the model dtype: kl_div(student, teacher) summed over the vocabulary
    (= KL(teacher || student)), plus the student's log-probability of each sampled token."""
    from torch.nn.functional import kl_div, log_softmax

    student = log_softmax((student_hidden @ weight.T) / temperature, dim=-1)
    teacher = log_softmax((teacher_hidden @ teacher_weight.T) / temperature, dim=-1)
    per_token = kl_div(student, teacher, reduction="none", log_target=True).sum(-1)
    return per_token, student.gather(-1, tokens[:, None]).squeeze(-1)


def sequence_forward_kl(model, student_prompt_ids: Sequence[int], teacher_hidden, completion_ids: Sequence[int], *,
                        chunk_tokens: int, temperature: float = 1.0, teacher_weight=None):
    """Per-token KL(teacher || student) over the full vocabulary, with gradients reaching only the student, and
    the student's detached log-probabilities of the completion tokens.

    `teacher_hidden` holds the teacher model's final-norm states on the same completion, computed beforehand under
    no_grad; `teacher_weight` is its LM head (default: the student's). Logits are materialised in recomputed chunks.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    if not completion_ids:
        raise ValueError("Student completion is empty.")
    weight = base_model(model).lm_head.weight
    teacher_weight = weight if teacher_weight is None else teacher_weight
    if weight.requires_grad:
        raise ValueError("The LM head must stay frozen.")
    student_hidden = completion_hidden(model, student_prompt_ids, completion_ids)
    if student_hidden.shape != teacher_hidden.shape:
        raise ValueError("Teacher and student states cover different completion positions.")
    tokens = torch.tensor(list(completion_ids), dtype=torch.long, device=student_hidden.device)
    per_token, logps = [], []
    for start in range(0, len(completion_ids), chunk_tokens):
        stop = start + chunk_tokens
        kl, selected = checkpoint(_chunk_forward_kl, student_hidden[start:stop], teacher_hidden[start:stop],
                                  weight, teacher_weight, tokens[start:stop], temperature, use_reentrant=False)
        per_token.append(kl)
        logps.append(selected.detach())
    return torch.cat(per_token), torch.cat(logps)


def sequence_loss(per_token_kl, student_logps, sampling_logps, *, skip: int, cap: float):
    """The training loss for one sequence (batch size 1).

    The first `skip` completion tokens are masked; with server rollouts the per-token loss is weighted by the
    sequence mean of the capped importance ratio exp(student - sampler); then the token mean over the unmasked
    completion. Returns the loss and the importance weight (None without server log-probabilities).
    """
    import torch

    per_token_loss = per_token_kl[None]
    completion_mask = torch.ones_like(per_token_loss, dtype=torch.long)
    loss_completion_mask = completion_mask
    if skip > 0:
        token_positions = torch.arange(per_token_loss.shape[1], device=per_token_loss.device)[None]
        loss_completion_mask = completion_mask * (token_positions >= skip).int()
    weight = None
    if sampling_logps is not None:
        ratio = torch.clamp(torch.exp(student_logps[None] - sampling_logps[None]), max=cap)
        importance_weights = (ratio * loss_completion_mask).sum(-1) / loss_completion_mask.sum(-1).clamp(min=1.0)
        per_token_loss = per_token_loss * importance_weights.unsqueeze(-1)
        weight = float(importance_weights[0])
    loss = ((per_token_loss * loss_completion_mask).sum(-1) / loss_completion_mask.sum(-1).clamp(min=1.0)).mean()
    return loss, weight


def reference_forward_kl(student_logits, teacher_logits):
    """Unchunked per-token objective, used to test the chunked path."""
    import torch
    from torch.nn.functional import kl_div, log_softmax

    return kl_div(log_softmax(student_logits, dim=-1), log_softmax(teacher_logits.detach(), dim=-1),
                  reduction="none", log_target=True).sum(-1)


# ---------------------------------------------------------------------------
# Stage training


@dataclass(frozen=True)
class StageSummary:
    stage: int
    task_id: str
    updates: int
    batch_size: int
    mean_loss: float
    final_loss: float
    truncated_prompts: list[str]
    seconds: float


@dataclass(frozen=True)
class StageProgress:
    """Resumable position inside a stage: the next update and what came before it."""
    next_update: int
    losses: tuple[float, ...]
    seconds: float


def train_stage(model, tokenizer, *, stage: int, task_id: str, records: Sequence[Mapping[str, Any]],
                updates: int, batch_size: int, learning_rate: float, gradient_clip: float, seed: int,
                eos: Sequence[int], config: SdftConfig, teacher: EmaTeacher,
                log: Callable[[dict[str, Any]], None], progress: StageProgress | None = None,
                optimizer_state: Any = None, checkpoint_every: int = 0,
                save_checkpoint: Callable[[StageProgress, Any], None] | None = None,
                sampler: Callable[..., list[Rollout]] | None = None) -> StageSummary:
    """Fixed-update SDFT on one stage; the caller merges the LoRA afterwards. `sampler` replaces the in-process
    rollout generator with the same signature (e.g. an inference server serving the current student). A resumed
    stage expects the caller to have restored the RNG state saved with its checkpoint."""
    import torch
    from transformers import set_seed

    started = time.perf_counter()
    if progress is None:
        set_seed(seed)  # as Trainer.__init__
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    teacher_weight = base_model(teacher.model).lm_head.weight
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    student_inputs = [prompt_ids(tokenizer, str(r["prompt"]), config.max_prompt_length) for r in records]
    teacher_inputs = [prompt_ids(tokenizer, teacher_prompt(r), config.max_prompt_length) for r in records]
    truncated = [str(r["id"]) for r, s, t in zip(records, student_inputs, teacher_inputs) if s[1] or t[1]]
    schedule = update_batches(len(records), updates=updates, batch_size=batch_size, seed=seed)
    first = progress.next_update if progress else 0
    losses = list(progress.losses) if progress else []
    prior = progress.seconds if progress else 0.0
    for update in range(first, updates):
        indices = schedule[update]
        student_ids = [student_inputs[index][0] for index in indices]
        teacher_ids = [teacher_inputs[index][0] for index in indices]
        update_started = time.perf_counter()
        model.eval()
        rollouts = (sampler or sample_rollouts)(model, tokenizer, student_ids, config=config, eos=eos)
        with torch.no_grad():
            teacher_hidden = [completion_hidden(teacher.model, ids, list(rollout.token_ids))
                              for ids, rollout in zip(teacher_ids, rollouts)]
        generation_seconds = time.perf_counter() - update_started
        model.train()
        rate = learning_rate_at(learning_rate, update, updates, config.warmup_ratio)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        sample_losses = []
        for index, student, rollout, hidden, ids in zip(indices, student_ids, rollouts, teacher_hidden, teacher_ids):
            per_token, logps = sequence_forward_kl(model, student, hidden, list(rollout.token_ids),
                                                   chunk_tokens=config.kl_chunk_tokens,
                                                   temperature=config.rollout_temperature,
                                                   teacher_weight=teacher_weight)
            sampling = (torch.tensor(rollout.sampling_logprobs, dtype=torch.float32, device=logps.device)
                        if rollout.sampling_logprobs is not None else None)
            loss, weight = sequence_loss(per_token, logps, sampling, skip=config.num_loss_tokens_to_skip,
                                         cap=config.importance_sampling_cap)
            (loss / batch_size).backward()
            sample_losses.append(float(loss.detach()))
            log({"kind": "sample", "stage": stage, "update": update, "id": str(records[index]["id"]),
                 "loss": sample_losses[-1], "importanceWeight": weight,
                 "lossTokens": max(0, len(rollout.token_ids) - config.num_loss_tokens_to_skip),
                 "rolloutTokens": len(rollout.token_ids), "rolloutStopReason": rollout.stop_reason,
                 "rollout": rollout.text, "teacherPromptTokens": len(ids)})
        del teacher_hidden
        norm = float(torch.nn.utils.clip_grad_norm_(trainable, gradient_clip))
        optimizer.step()
        teacher.update()
        losses.append(sum(sample_losses) / len(sample_losses))
        log({"kind": "update", "stage": stage, "update": update, "loss": losses[-1], "gradNorm": norm,
             "learningRate": rate, "generationSeconds": generation_seconds,
             "seconds": time.perf_counter() - update_started,
             "cudaPeakBytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None})
        done = update + 1
        if save_checkpoint is not None and checkpoint_every and done % checkpoint_every == 0:
            save_checkpoint(StageProgress(done, tuple(losses), prior + time.perf_counter() - started),
                            optimizer.state_dict())
    return StageSummary(stage, task_id, updates, batch_size, sum(losses) / len(losses), losses[-1],
                        truncated, prior + time.perf_counter() - started)


def summary_dict(summary: StageSummary) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(summary)))
