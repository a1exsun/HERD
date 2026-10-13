"""TreeLoRA (ICML 2025) on the shared TRACE+17 protocol.

`upstream/utils/kd_lora_tree.py` (gradient-similarity tree, bandit search and regularizer) is executed
unchanged. This module re-hosts the training step of `upstream/model/Regular/Tree_LoRA.py`
(`train_one_task`) without DeepSpeed, running on one GPU with gradient accumulation:

- every micro-step is one record; per micro-step, in order: `kd_lora_tree.step()`, the record's loss,
  the stacked LoRA-A rows, `insert_grad`, and from the second task on `tree_search` + `get_loss` with
  `loss = loss - reg_loss`; the optimizer steps once per accumulation window (32 records);
- `new_epoch_init` at the start of each pass over the stage's records and `end_task` after the stage;
- LoRA r = 8, alpha = 32, dropout 0.1 (`training/main.py`), reg = 0.5, learning rate 1e-4
  (`scripts/lora_based_methods/Tree_LoRA.sh`), Adam with betas (0.9, 0.95) and no weight decay at a
  constant rate (`get_optimizer`), gradient clipping 1.0 (`utils/ds_utils.py`), one optimizer state for
  the whole stream and one LoRA trained through all tasks.

torch is imported lazily.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys
import time
import types
from typing import Any, Callable, Mapping, Sequence

from microbatch import backward_records
from sft import StageSummary, TrainingProgress, batch_schedule, supervised_token_ids

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "upstream"


@dataclass(frozen=True)
class TreeLoraConfig:
    # training/main.py (Tree_LoRA branch and get_optimizer), scripts/lora_based_methods/Tree_LoRA.sh,
    # utils/ds_utils.py.
    rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    reg: float = 0.5
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.0
    gradient_clip: float = 1.0


def print_rank_0(msg, rank=0):
    # upstream utils/utils.py, verbatim.
    if rank <= 0:
        print(msg)


@lru_cache(maxsize=1)
def upstream_tree():
    """Execute upstream `utils/kd_lora_tree.py` unchanged.

    Its one project import, `from utils.utils import print_rank_0`, names a module that imports DeepSpeed at
    import time; the same function is registered under that name for the import only."""
    shim = types.ModuleType("utils.utils")
    shim.print_rank_0 = print_rank_0
    package = types.ModuleType("utils")
    package.__path__ = []
    package.utils = shim
    saved = {name: sys.modules.get(name) for name in ("utils", "utils.utils")}
    sys.modules.update({"utils": package, "utils.utils": shim})
    try:
        spec = importlib.util.spec_from_file_location("treelora_upstream_kd_lora_tree",
                                                      UPSTREAM / "utils/kd_lora_tree.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


# ---------------------------------------------------------------------------
# Model, optimizer and the KD tree's persistent state


def trainable_parameters(model) -> list[Any]:
    """Name-sorted, so a saved optimizer state maps onto a reloaded adapter."""
    return [parameter for _, parameter in sorted(model.named_parameters()) if parameter.requires_grad]


def make_optimizer(model, config: TreeLoraConfig):
    import torch

    return torch.optim.Adam(trainable_parameters(model), lr=config.learning_rate,
                            betas=(config.adam_beta1, config.adam_beta2), weight_decay=config.weight_decay)


def stacked_lora_a(model):
    """Tree_LoRA.train_one_task lines 78-87: each LoRA module's A matrix, flattened, one row per module.

    Upstream filters `"loranew_A"` (its forked peft); standard peft names the same matrix `lora_A`. Upstream
    stacks equal-width rows (q_proj/v_proj); the shared target set has several input widths, so rows are
    zero-padded to the widest, which leaves every L1 distance, inner product and tree split unchanged."""
    import torch

    rows = [parameter.reshape(-1) for name, parameter in model.named_parameters() if "lora_A" in name]
    if not rows:
        raise ValueError("No LoRA-A parameters found.")
    width = max(row.numel() for row in rows)
    return torch.stack([torch.nn.functional.pad(row, (0, width - row.numel())) for row in rows], dim=0)


def new_tree(num_tasks: int, reg: float, fingerprints: Sequence[Any]):
    """The stream's `KD_LoRA_Tree` after `end_task` of every sealed task, as it stands upstream in memory.

    `fingerprints[k]` is task k's `current_grad` at its `end_task`; `end_task` of the last one rebuilds the
    tree from all of them exactly as upstream did after that task."""
    tree = upstream_tree().KD_LoRA_Tree(types.SimpleNamespace(reg=reg, num_tasks=num_tasks, global_rank=0))
    if fingerprints:
        last = len(fingerprints) - 1
        for task_id, fingerprint in enumerate(fingerprints[:last]):
            tree.all_accumulate_grads[task_id] = fingerprint
        tree.current_grad = fingerprints[last]
        tree.end_task(task_id=last)
    return tree


_EPOCH_FIELDS = ("current_grad", "sim", "num_of_selected", "tmp_rounds", "total_rounds", "tmp_reg")


def tree_epoch_state(tree) -> dict[str, Any]:
    """The tree's within-epoch state at an update boundary (mid-stage checkpoint)."""
    state = {name: getattr(tree, name, None) for name in _EPOCH_FIELDS}
    state["all_grad_loaded"] = tree.all_grad is not None
    return state


def restore_tree_epoch_state(tree, state: Mapping[str, Any], task_id: int, device) -> None:
    import torch

    for name in _EPOCH_FIELDS:
        setattr(tree, name, state[name])
    if state["all_grad_loaded"]:  # as the first tree_search of an epoch sets it (kd_lora_tree.py:197-198)
        tree.all_grad = torch.stack(tree.all_accumulate_grads[:task_id], dim=0).to(device, non_blocking=True)
        tree.all_grad_device = tree.all_grad


def tree_dict(node) -> dict[str, Any] | None:
    if node is None:
        return None
    return {"taskIndices": list(node.task_indices), "depth": node.depth, "isLeaf": node.is_leaf,
            "medianSimilarity": node.median_similarity,
            "left": tree_dict(node.left), "right": tree_dict(node.right)}


# ---------------------------------------------------------------------------
# Stage training


def epoch_lengths(records: int, micro_steps: int) -> list[int]:
    """Micro-steps per epoch: one pass over the records each; a final partial pass joins the last epoch."""
    if records < 1 or micro_steps < 1:
        raise ValueError("records and micro_steps must be positive.")
    passes, remainder = divmod(micro_steps, records)
    if passes == 0:
        return [micro_steps]
    lengths = [records] * passes
    lengths[-1] += remainder
    return lengths


def train_stage(model, tokenizer, optimizer, tree, records: Sequence[Mapping[str, Any]], *, task_id: int,
                updates: int, batch_size: int, config: TreeLoraConfig, seed: int, max_length: int,
                selection, log: Callable[[dict[str, Any]], None], resume: TrainingProgress | None = None,
                checkpoint_every: int = 0,
                checkpoint: Callable[[TrainingProgress], None] | None = None) -> StageSummary:
    """Upstream `Tree_LoRA.train_one_task` for one task with `batch_size` micro-steps per optimizer step.

    `selection` (tasks x LoRA modules, or None at the first task) counts the tasks `tree_search` picked."""
    import torch

    if not records:
        raise ValueError("Stage training records must be non-empty.")
    device = next(model.parameters()).device
    encoded = [supervised_token_ids(tokenizer, record, max_length=max_length) for record in records]
    schedule = batch_schedule(len(encoded), updates=updates, batch_size=batch_size, seed=seed)
    order = [index for batch in schedule for index in batch]
    epoch_starts, start = {}, 0
    for length in epoch_lengths(len(encoded), len(order)):
        epoch_starts[start] = length
        start += length
    trainable = trainable_parameters(model)
    progress = resume or TrainingProgress(0, (), 0, 0.0)
    losses, total_tokens = list(progress.losses), progress.target_tokens
    started = time.perf_counter()
    model.train()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for update in range(progress.update, updates):
        optimizer.zero_grad(set_to_none=True)
        first = update * batch_size
        batch = [encoded[order[first + k]] for k in range(batch_size)]
        # The window's records' cross-entropy, backpropagated as sum_k CE_k / batch_size (DeepSpeed scales each
        # micro-step's loss by 1 / gradient_accumulation_steps).
        record_losses = backward_records(model, batch, batch_size=batch_size, pad_id=pad_id)
        # The parameters are fixed inside the window, so the stacked LoRA-A rows are the same at every micro-step.
        _grad_current = stacked_lora_a(model)
        regularization = []
        for k in range(batch_size):
            if first + k in epoch_starts:
                tree.new_epoch_init(epoch_starts[first + k])
            if config.reg > 0:
                tree.step()
            loss = record_losses[k]
            if config.reg > 0:
                tree.insert_grad(_grad_current)
                if task_id > 0:
                    prev_id_matrix = tree.tree_search(task_id, device=device)
                    reg_loss = tree.get_loss(_grad_current, loss, task_id, prev_id_matrix)
                    regularization.append(reg_loss)
                    selection[prev_id_matrix, torch.arange(prev_id_matrix.numel(), device=prev_id_matrix.device)] += 1
                    if tree.tmp_rounds % 100 == 0:
                        print_rank_0("\033[34m(Normal Process) Sim: {};\033[0m".format(tree.sim))
                        print_rank_0("\033[34m(Normal Process) Selected Nums: {};\033[0m".format(
                            tree.num_of_selected[:task_id]))
                        print_rank_0("\033[34m(Normal Process) Prev_id_matrix: {};\033[0m".format(prev_id_matrix))
                        print_rank_0("\033[34mReg Loss: {:.4f}\033[0m".format(reg_loss))
        task_loss = sum(float(loss) for loss in record_losses) / batch_size
        reg_mean = 0.0
        if regularization:
            reg_sum = torch.stack(regularization).sum()
            (-reg_sum / batch_size).backward()  # loss = loss - reg_loss at every micro-step
            reg_mean = float(reg_sum.detach()) / batch_size
        tokens = sum(sum(label != -100 for label in labels) for _, labels in batch)
        norm = torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip)
        optimizer.step()
        losses.append(task_loss - reg_mean)
        total_tokens += tokens
        log({"update": update, "loss": task_loss - reg_mean, "taskLoss": task_loss, "regLoss": reg_mean,
             "tokens": tokens, "gradNorm": float(norm)})
        done = update + 1
        if checkpoint is not None and checkpoint_every and done % checkpoint_every == 0 and done < updates:
            checkpoint(TrainingProgress(done, tuple(losses), total_tokens,
                                        progress.seconds + time.perf_counter() - started))
    return StageSummary(updates, batch_size, len(records), sum(losses) / len(losses), losses[-1], total_tokens,
                        progress.seconds + time.perf_counter() - started)
