"""MoRAM continual learning (ICML 2026, arXiv:2506.21035) on the shared TRACE+17 protocol.

This module runs the layer code in `upstream/` and re-hosts the TRACE trainer
(`model/Regular/MoRAM.py`
and the MoRAM branch of `training/main.py`) without DeepSpeed or the forked peft
package:

- every target Linear becomes an upstream `MoRAMLinear` sharing the frozen base
  weight, with rank r new rank-1 atoms per task, lora_alpha = 2r, dropout 0.1,
  top-k = r, router temperature and inference threshold from `runner_moram.sh`;
- only the current task's atoms (`loranew_*`) train; all earlier atoms are frozen;
- the loss is the per-sequence token-mean cross-entropy, averaged over the batch;
  Adam with betas (0.9, 0.95), no weight decay, a constant learning rate and
  gradient clipping 1.0; one optimizer state for the whole stream;
- after each task, `increment_task` appends the trained atoms to the frozen bank and
  re-initializes the new atoms, as `save_model` does before
  writing the checkpoint that inference then loads.

torch is imported lazily so the pure schedule logic is testable without it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import random
import time
from typing import Any, Callable, Mapping, Sequence

from microbatch import backward_records


ADAPTER = "default"


@dataclass(frozen=True)
class MoramConfig:
    # TRACE/runner_moram.sh and the MoRAM branch of TRACE/training/main.py.
    rank: int = 8                        # moram_rank: new rank-1 atoms per task
    topk: int = 8                        # moram_topk = moram_rank
    router_temperature: float = 0.03     # moram_router_temp
    infer_threshold: float = 0.2         # moram_infer_lora_a_thresh (inference only)
    lora_alpha: int = 16                 # max(1, 2 * moram_rank)
    lora_dropout: float = 0.1
    learning_rate: float = 5e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.0
    gradient_clip: float = 1.0           # DeepSpeed gradient_clipping


def moram_targets(model) -> list[str]:
    """The shared text-backbone Linear modules; vision, MTP and the LM head stay unadapted."""
    import torch

    excluded = ("visual", "vision", "mtp", "lm_head")
    targets = sorted(name for name, module in model.named_modules()
                     if isinstance(module, torch.nn.Linear) and not any(part in name.lower() for part in excluded))
    if not targets:
        raise ValueError("No text-backbone Linear modules found.")
    return targets


def moram_modules(model) -> list[tuple[str, Any]]:
    from upstream.moram import MoRAMLinear

    return [(name, module) for name, module in model.named_modules() if isinstance(module, MoRAMLinear)]


def inject_moram(model, config: MoramConfig, targets: Sequence[str]) -> None:
    """Replace each target Linear with an upstream MoRAMLinear that shares its frozen weight and bias.

    Mirrors upstream `LoraModel._find_and_replace` / `_replace_module` for the MoRAM branch.
    """
    from upstream.moram import MoRAMLinear

    modules = dict(model.named_modules())
    for name in targets:
        old = modules[name]
        parent_name, _, child = name.rpartition(".")
        parent = modules[parent_name] if parent_name else model
        new = MoRAMLinear(ADAPTER, old.in_features, old.out_features, r=config.rank, lora_alpha=config.lora_alpha,
                          lora_dropout=config.lora_dropout, fan_in_fan_out=False, bias=old.bias is not None,
                          r_sum=0, moram_topk=config.topk, device=old.weight.device, dtype=old.weight.dtype)
        setattr(parent, child, new)
        new.weight = old.weight
        if old.bias is not None:
            new.bias = old.bias
        for sub_name, submodule in new.named_modules():
            if "lora_" in sub_name:
                submodule.to(old.weight.device)
        new.set_router_temperature(config.router_temperature)
        new.set_moram_infer_lora_a_threshold(config.infer_threshold)
    mark_trainable(model)


def mark_trainable(model) -> None:
    """Upstream rule: only `loranew_*` (the current task's atoms) receive gradients."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "loranew_" in name


def trainable_parameters(model) -> list[Any]:
    return [parameter for _, parameter in sorted(model.named_parameters()) if parameter.requires_grad]


def make_optimizer(model, config: MoramConfig):
    import torch

    return torch.optim.Adam(trainable_parameters(model), lr=config.learning_rate,
                            betas=(config.adam_beta1, config.adam_beta2), weight_decay=config.weight_decay)


def increment_all(model, *, seed: int) -> None:
    """Upstream `save_model`: consolidate the trained atoms into the frozen bank, re-initialize new atoms."""
    import torch

    torch.manual_seed(seed)  # the re-initialization draws are part of the sealed state
    for _, module in moram_modules(model):
        module.increment_task()
    mark_trainable(model)


def moram_state(model) -> dict[str, Any]:
    state = {}
    for name, module in moram_modules(model):
        for part in ("lora_A", "lora_B", "loranew_A", "loranew_B"):
            state[f"{name}.{part}"] = getattr(module, part)[ADAPTER].weight.detach().cpu().contiguous()
    return state


def trainable_atom_state(model) -> dict[str, Any]:
    """Only current-task atoms change during training; the sealed prior bank is the recovery anchor."""
    return {f"{name}.{part}": getattr(module, part)[ADAPTER].weight.detach().cpu().contiguous()
            for name, module in moram_modules(model) for part in ("loranew_A", "loranew_B")}


def load_trainable_atom_state(model, state: Mapping[str, Any]) -> None:
    import torch

    parameters = {f"{name}.{part}": getattr(module, part)[ADAPTER].weight
                  for name, module in moram_modules(model) for part in ("loranew_A", "loranew_B")}
    if set(parameters) != set(state):
        raise ValueError("Trainable atom checkpoint does not match the injected modules.")
    for name, parameter in parameters.items():
        if parameter.shape != state[name].shape or parameter.dtype != state[name].dtype:
            raise ValueError(f"Trainable atom checkpoint shape or dtype mismatch: {name}")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state[name].to(parameter.device))


def load_moram_state(model, state: Mapping[str, Any]) -> None:
    """Restore every atom bank, resizing the frozen banks to the saved number of consolidated atoms."""
    import torch

    names = {f"{name}.{part}" for name, _ in moram_modules(model)
             for part in ("lora_A", "lora_B", "loranew_A", "loranew_B")}
    if names != set(state):
        raise ValueError("MoRAM state does not match the injected modules.")
    for name, module in moram_modules(model):
        device = module.loranew_A[ADAPTER].weight.device
        dtype = module.loranew_A[ADAPTER].weight.dtype
        old_a, old_b = state[f"{name}.lora_A"], state[f"{name}.lora_B"]
        bank_a = torch.nn.Linear(module.in_features, old_a.shape[0], bias=False).to(device=device, dtype=dtype)
        bank_b = torch.nn.Linear(old_b.shape[1], module.out_features, bias=False).to(device=device, dtype=dtype)
        with torch.no_grad():
            bank_a.weight.copy_(old_a)
            bank_b.weight.copy_(old_b)
            module.loranew_A[ADAPTER].weight.copy_(state[f"{name}.loranew_A"])
            module.loranew_B[ADAPTER].weight.copy_(state[f"{name}.loranew_B"])
        module.lora_A[ADAPTER] = bank_a
        module.lora_B[ADAPTER] = bank_b
    mark_trainable(model)


def atom_counts(model) -> dict[str, int]:
    """Consolidated and in-training atoms, and the stored atom parameters (the method's growing state)."""
    modules = moram_modules(model)
    consolidated = {module.lora_A[ADAPTER].out_features for _, module in modules}
    if len(consolidated) != 1:
        raise ValueError("MoRAM modules hold different numbers of consolidated atoms.")
    parameters = sum(module.lora_A[ADAPTER].weight.numel() + module.lora_B[ADAPTER].weight.numel()
                     + module.loranew_A[ADAPTER].weight.numel() + module.loranew_B[ADAPTER].weight.numel()
                     for _, module in modules)
    return {"modules": len(modules), "consolidatedAtomsPerModule": consolidated.pop(), "atomParameters": parameters}


# ---------------------------------------------------------------------------
# Supervised targets and stage training


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


def batch_schedule(size: int, *, updates: int, batch_size: int, seed: int) -> list[list[int]]:
    """Record indices per update: seeded shuffle, reshuffled at each epoch boundary (the upstream
    RandomSampler), deterministic so a resumed stage continues the same sequence."""
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
    update: int
    losses: tuple[float, ...]
    target_tokens: int
    seconds: float


def train_stage(model, tokenizer, optimizer, records: Sequence[Mapping[str, Any]], *, updates: int,
                batch_size: int, config: MoramConfig, seed: int, max_length: int,
                log: Callable[[dict[str, Any]], None], resume: TrainingProgress | None = None,
                checkpoint_every: int = 0,
                checkpoint: Callable[[TrainingProgress], None] | None = None) -> StageSummary:
    import torch

    if not records:
        raise ValueError("Stage training records must be non-empty.")
    device = next(model.parameters()).device
    encoded = [supervised_ids(tokenizer, record, max_length=max_length) for record in records]
    schedule = batch_schedule(len(encoded), updates=updates, batch_size=batch_size, seed=seed)
    trainable = trainable_parameters(model)
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
        norm = torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip)
        optimizer.step()
        losses.append(batch_loss)
        total_tokens += tokens
        log({"update": update, "loss": batch_loss, "tokens": tokens, "gradNorm": float(norm)})
        done = update + 1
        if checkpoint is not None and checkpoint_every and done % checkpoint_every == 0:
            checkpoint(TrainingProgress(done, tuple(losses), total_tokens,
                                        progress.seconds + time.perf_counter() - started))
    return StageSummary(updates, batch_size, len(records), sum(losses) / len(losses), losses[-1], total_tokens,
                        progress.seconds + time.perf_counter() - started)


def summary_dict(summary: StageSummary) -> dict[str, Any]:
    return asdict(summary)
