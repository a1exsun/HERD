from __future__ import annotations

from dataclasses import asdict, dataclass
import random
import time
from typing import Any, Callable, Mapping, Sequence

from microbatch import backward_records


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


def _require_target(record: Mapping[str, Any]) -> str:
    answer = record.get("answer")
    if isinstance(answer, str) and answer:
        return answer
    references = [value for value in record.get("references", []) if isinstance(value, str) and value]
    if references:
        return references[0]
    raise ValueError(f"Record {record.get('id')} has no supervised target.")


def _template_ids(tokenizer, messages, *, add_generation_prompt: bool) -> list[int]:
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt,
                                            enable_thinking=False, return_dict=True)
    return list(encoded["input_ids"])


def supervised_token_ids(tokenizer, record: Mapping[str, Any], *, max_length: int = 16384) -> tuple[list[int], list[int]]:
    prompt = str(record["prompt"])
    answer = _require_target(record)
    prompt_ids = _template_ids(tokenizer, [{"role": "user", "content": prompt}], add_generation_prompt=True)
    full_ids = _template_ids(tokenizer, [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
                             add_generation_prompt=False)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenizer chat template does not preserve the supervised prompt prefix.")
    if len(full_ids) > max_length:
        raise ValueError(f"Supervised example exceeds {max_length} tokens; silent truncation is disabled.")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if all(label == -100 for label in labels):
        raise ValueError("Supervised record has no assistant tokens after chat templating.")
    return list(full_ids), labels


def text_backbone_lora_targets(model) -> list[str]:
    import torch

    excluded = ("visual", "vision", "mtp", "lm_head")
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and not any(part in name.lower() for part in excluded)
    ]
    if not targets:
        raise ValueError("No text-backbone Linear modules were found for LoRA.")
    return sorted(targets)


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


def train_stage(model, tokenizer, records: Sequence[Mapping[str, Any]], *, updates: int, batch_size: int,
                learning_rate: float, gradient_clip: float, seed: int, max_length: int,
                log: Callable[[dict[str, Any]], None], resume: TrainingProgress | None = None,
                optimizer_state: Mapping[str, Any] | None = None, checkpoint_every: int = 0,
                checkpoint: Callable[[TrainingProgress, Any], None] | None = None) -> StageSummary:
    """Current-task SFT: per-record token-mean cross-entropy averaged over the batch, AdamW at a constant
    learning rate, one optimizer per stage. Records are processed in right-padded micro-batches and only the
    answer positions are projected to the vocabulary."""
    import torch

    if not records:
        raise ValueError("Current stage training records must be non-empty.")
    device = next(model.parameters()).device
    encoded = [supervised_token_ids(tokenizer, record, max_length=max_length) for record in records]
    schedule = batch_schedule(len(encoded), updates=updates, batch_size=batch_size, seed=seed)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
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
        batch = [encoded[index] for index in schedule[update]]
        record_losses = backward_records(model, batch, batch_size=batch_size, pad_id=pad_id)
        batch_loss = sum(float(loss) for loss in record_losses) / batch_size
        tokens = sum(sum(label != -100 for label in labels) for _, labels in batch)
        norm = torch.nn.utils.clip_grad_norm_(trainable, gradient_clip)
        optimizer.step()
        losses.append(batch_loss)
        total_tokens += tokens
        log({"update": update, "loss": batch_loss, "tokens": tokens, "gradNorm": float(norm)})
        done = update + 1
        if checkpoint is not None and checkpoint_every and done % checkpoint_every == 0 and done < updates:
            checkpoint(TrainingProgress(done, tuple(losses), total_tokens,
                                        progress.seconds + time.perf_counter() - started), optimizer)
    return StageSummary(updates, batch_size, len(records), sum(losses) / len(losses), losses[-1], total_tokens,
                        progress.seconds + time.perf_counter() - started)


def summary_dict(summary: StageSummary) -> dict[str, Any]:
    return asdict(summary)
