"""Uniform experience replay with a fixed record-count reservoir, self-contained.

The learner carries one LoRA across the stream and R_t = G union R_t^task:
- G is the fixed, non-test general set, preserved verbatim;
- R_t^task holds the k smallest fixed SHA-256("<seed>:<id>") priorities among
  trainable task records seen so far, irrespective of their serialized lengths.

Under uniform random priorities this is a simple random sample of min(k, N_t)
records: each historical record has inclusion probability min(1, k / N_t).
SHA-256 supplies deterministic pseudorandom priorities. A rejected key cannot
enter a later bottom-k set, so only the retained records and new data are read.
No record is truncated or filtered for memory length. Actual token storage is
reported separately; this control does not claim token-budget equality to HERO.

Stage t trains on D_t union R_(t-1)^task union G. The incoming task updates the
reservoir only after that stage's training pool is assembled. No teacher,
auxiliary loss or harness is used. Model dependencies are imported lazily.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import random
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from microbatch import backward_records


# ---------------------------------------------------------------------------
# Serialization and token accounting (HERO's counting rule)


def serialize_record(record: Mapping[str, Any]) -> str:
    """One compact JSON line, as JavaScript's JSON.stringify writes it, plus the newline."""
    return json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"


def count_serialized_tokens(tokenizer, record: Mapping[str, Any]) -> int:
    """Tokenizer tokens of the complete serialized record, without special tokens."""
    return len(tokenizer.encode(serialize_record(record), add_special_tokens=False))


def selection_key(seed: int, identifier: str) -> str:
    return hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MemoryRecord:
    """A retained record, its source task, the stage that introduced it and its serialized size."""
    record: dict[str, Any]
    task_id: str
    introduced_at: int
    tokens: int

    @property
    def id(self) -> str:
        return str(self.record["id"])


@dataclass
class UniformReplayMemory:
    """Fixed-count bottom-k reservoir; record length never influences selection."""
    capacity: int
    seed: int
    records: list[MemoryRecord] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("Replay record capacity must be positive.")

    def update(self, incoming: Iterable[MemoryRecord]) -> None:
        """Consume newly seen records, keeping a length-independent bottom-k set."""
        unique = {item.id: item for item in self.records}
        for candidate in incoming:
            if candidate.tokens <= 0:
                raise ValueError("Serialized record token counts must be positive.")
            previous = unique.get(candidate.id)
            if previous is not None and previous != candidate:
                raise ValueError(f"Conflicting records share the stable id {candidate.id}.")
            unique.setdefault(candidate.id, candidate)
        self.records = sorted(unique.values(),
                              key=lambda item: (selection_key(self.seed, item.id), item.id))[:self.capacity]


def pack_fixed_general(records: Sequence[MemoryRecord], *, budget: int) -> list[MemoryRecord]:
    """The reserved general set is kept whole; it must fit its own reservation (never borrowed)."""
    total = sum(item.tokens for item in records)
    if total > budget:
        raise ValueError(f"General records need {total} tokens, above their {budget}-token reservation.")
    return list(records)


def memory_summary(records: Sequence[MemoryRecord], stage: int) -> dict[str, Any]:
    by_task: dict[str, dict[str, Any]] = {}
    for item in records:
        row = by_task.setdefault(item.task_id, {"records": 0, "tokens": 0, "minimumAge": None, "maximumAge": None})
        age = stage - item.introduced_at
        row["records"] += 1
        row["tokens"] += item.tokens
        row["minimumAge"] = age if row["minimumAge"] is None else min(row["minimumAge"], age)
        row["maximumAge"] = age if row["maximumAge"] is None else max(row["maximumAge"], age)
    return {"records": len(records), "tokens": sum(item.tokens for item in records), "byTask": by_task,
            "ids": [item.id for item in records]}


def training_pool(current: Sequence[Mapping[str, Any]], buffer: Sequence[MemoryRecord],
                  general: Sequence[MemoryRecord]) -> list[dict[str, Any]]:
    """D_t union R_(t-1)^task union G, each row tagged with its source for the training log."""
    pool = [{**record, "replaySource": "current"} for record in current]
    pool += [{**item.record, "replaySource": "buffer"} for item in buffer]
    pool += [{**item.record, "replaySource": "general"} for item in general]
    identifiers = [str(row["id"]) for row in pool]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("A record appears twice in one stage's training pool.")
    return pool


# ---------------------------------------------------------------------------
# Supervised targets (chat template, prompt masked, no silent truncation)


def reference_target(record: Mapping[str, Any]) -> str:
    answer = record.get("answer")
    if isinstance(answer, str) and answer:
        return answer
    references = [value for value in record.get("references", []) if isinstance(value, str) and value]
    if references:
        return references[0]
    raise ValueError(f"Record {record.get('id')} has no supervised target.")


def chat_ids(tokenizer, prompt: str) -> list[int]:
    return list(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                              add_generation_prompt=True, enable_thinking=False,
                                              return_dict=True)["input_ids"])


def supervised_ids(tokenizer, record: Mapping[str, Any], *, max_length: int) -> tuple[list[int], list[int]]:
    """Chat-templated (prompt, first reference) with the prompt masked; raises instead of truncating."""
    prompt = str(record["prompt"])
    prompt_ids = chat_ids(tokenizer, prompt)
    full_ids = list(tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": reference_target(record)}],
        tokenize=True, add_generation_prompt=False, enable_thinking=False, return_dict=True)["input_ids"])
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Chat template does not preserve the prompt prefix.")
    if len(full_ids) > max_length:
        raise ValueError(f"Supervised sequence exceeds {max_length} tokens.")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    if all(label == -100 for label in labels):
        raise ValueError("Supervised record has no assistant tokens after chat templating.")
    return full_ids, labels


def lora_targets(model) -> list[str]:
    """Text-backbone Linear modules; vision, MTP and the LM head stay frozen."""
    import torch

    excluded = ("visual", "vision", "mtp", "lm_head")
    targets = sorted(name for name, module in model.named_modules()
                     if isinstance(module, torch.nn.Linear) and not any(part in name.lower() for part in excluded))
    if not targets:
        raise ValueError("No text-backbone Linear modules found.")
    return targets


# ---------------------------------------------------------------------------
# Stage training (lazy torch)


def batch_schedule(size: int, *, updates: int, batch_size: int, seed: int) -> list[list[int]]:
    """Record indices per update: seeded shuffle, reshuffled at each epoch boundary. Deterministic, so a
    resumed stage continues the same sequence."""
    order = list(range(size))
    rng = random.Random(seed)
    rng.shuffle(order)
    schedule, cursor = [], 0
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


@dataclass(frozen=True)
class StageSummary:
    updates: int
    batch_size: int
    records: int
    mean_loss: float
    final_loss: float
    target_tokens: int
    seconds: float


@dataclass(frozen=True)
class TrainingProgress:
    """State after `update` completed updates; with the adapter and optimizer it resumes a stage exactly."""
    update: int
    losses: tuple[float, ...]
    target_tokens: int
    seconds: float


def train_stage(model, tokenizer, records: Sequence[Mapping[str, Any]], *, updates: int, batch_size: int,
                learning_rate: float, gradient_clip: float, seed: int, max_length: int,
                log: Callable[[dict[str, Any]], None], resume: TrainingProgress | None = None,
                optimizer_state: Mapping[str, Any] | None = None, checkpoint_every: int = 0,
                checkpoint: Callable[[TrainingProgress, Any], None] | None = None) -> StageSummary:
    """The sequential-LoRA objective on the replay pool: per-record token-mean cross-entropy, averaged over
    the batch; AdamW at a constant learning rate; one optimizer per stage."""
    import torch

    if not records:
        raise ValueError("Stage training records must be non-empty.")
    device = next(model.parameters()).device
    encoded = [supervised_ids(tokenizer, record, max_length=max_length) for record in records]
    schedule = batch_schedule(len(encoded), updates=updates, batch_size=batch_size, seed=seed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    progress = resume or TrainingProgress(0, (), 0, 0.0)
    losses, total_tokens = list(progress.losses), progress.target_tokens
    started = time.perf_counter()
    model.train()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for update in range(progress.update, updates):
        optimizer.zero_grad(set_to_none=True)
        sources: dict[str, int] = {}
        for index in schedule[update]:
            source = str(records[index].get("replaySource", "current"))
            sources[source] = sources.get(source, 0) + 1
        batch = [encoded[index] for index in schedule[update]]
        record_losses = backward_records(model, batch, batch_size=batch_size, pad_id=pad_id)
        batch_loss = sum(float(loss) for loss in record_losses) / batch_size
        tokens = sum(sum(label != -100 for label in labels) for _, labels in batch)
        norm = torch.nn.utils.clip_grad_norm_(trainable, gradient_clip)
        optimizer.step()
        losses.append(batch_loss)
        total_tokens += tokens
        log({"update": update, "loss": batch_loss, "tokens": tokens, "gradNorm": float(norm), "sources": sources})
        done = update + 1
        if checkpoint is not None and checkpoint_every and done % checkpoint_every == 0 and done < updates:
            checkpoint(TrainingProgress(done, tuple(losses), total_tokens,
                                        progress.seconds + time.perf_counter() - started), optimizer)
    return StageSummary(updates, batch_size, len(records), sum(losses) / len(losses), losses[-1], total_tokens,
                        progress.seconds + time.perf_counter() - started)


def summary_dict(summary: StageSummary) -> dict[str, Any]:
    return asdict(summary)
