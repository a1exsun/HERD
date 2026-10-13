import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

from treelora_cl import (
    TreeLoraConfig, epoch_lengths, make_optimizer, new_tree, restore_tree_epoch_state, stacked_lora_a,
    train_stage, tree_epoch_state, upstream_tree,
)

PACKAGE = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        prompt = [1 + (ord(char) % 40) for char in messages[0]["content"]]
        ids = prompt if add_generation_prompt else prompt + [1 + (ord(char) % 40) for char in messages[-1]["content"]]
        return {"input_ids": ids}


RECORDS = [{"id": str(i), "prompt": f"question {i} " + "x" * i, "answer": f"answer {i * 7}"} for i in range(6)]


def tiny_lora_model(seed: int = 0):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    base = LlamaForCausalLM(LlamaConfig(vocab_size=48, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                                        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64))
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return get_peft_model(base, LoraConfig(r=4, lora_alpha=16, lora_dropout=0.0, target_modules=targets))


def lora_state(model):
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters() if "lora_" in name}


def test_upstream_files_are_unchanged():
    manifest = json.loads((PACKAGE / "upstream/UPSTREAM.json").read_text())
    assert manifest["commit"] == "1c7260c42b34e1961283797c742f08b9c3842501"
    assert "utils/kd_lora_tree.py" in manifest["files"]
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((PACKAGE / "upstream" / name).read_bytes()).hexdigest() == digest, name


def test_upstream_tree_is_loaded_without_leaving_a_utils_module():
    module = upstream_tree()
    assert Path(module.__file__).resolve() == PACKAGE / "upstream/utils/kd_lora_tree.py"
    assert {"KD_LoRA_Tree", "KDTreeNode", "tree_lora_loss"} <= set(vars(module))
    assert "utils" not in sys.modules and "utils.utils" not in sys.modules


def test_epochs_are_passes_over_the_records_with_a_partial_pass_joining_the_last():
    assert epoch_lengths(500, 1024) == [500, 524]
    assert epoch_lengths(256, 1024) == [256, 256, 256, 256]
    assert epoch_lengths(2000, 1024) == [1024]
    assert epoch_lengths(6, 12) == [6, 6]


def test_lora_rows_are_zero_padded_without_changing_distances():
    import torch

    model = tiny_lora_model()
    rows = stacked_lora_a(model)
    a = [parameter.reshape(-1) for name, parameter in model.named_parameters() if "lora_A" in name]
    assert rows.shape == (len(a), max(row.numel() for row in a))
    assert len({row.numel() for row in a}) > 1  # down_proj is wider
    for index, row in enumerate(a):
        assert torch.equal(rows[index, :row.numel()], row)
        assert not rows[index, row.numel():].any()


def reference_stream(model, tasks, *, config, batch_size, updates, seed):
    """A literal transcription of upstream Tree_LoRA.train_one_task/train_continual on one device with
    per_device_train_batch_size 1 and gradient_accumulation_steps = batch_size (DeepSpeed engine replaced by
    loss / gas backward, then clipping and an optimizer step every gas micro-steps)."""
    import torch

    from sft import batch_schedule, supervised_token_ids

    kd_lora_tree = upstream_tree().KD_LoRA_Tree(
        type("Args", (), {"reg": config.reg, "num_tasks": len(tasks), "global_rank": 0})())
    trainable = [p for _, p in sorted(model.named_parameters()) if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=config.learning_rate, betas=(config.adam_beta1, config.adam_beta2),
                                 weight_decay=config.weight_decay)
    fingerprints = []
    for task_id, records in enumerate(tasks):
        encoded = [supervised_token_ids(FakeTokenizer(), r, max_length=10_000) for r in records]
        order = [i for batch in batch_schedule(len(encoded), updates=updates, batch_size=batch_size,
                                               seed=seed + task_id) for i in batch]
        epochs = [order[start:start + len(encoded)] for start in range(0, len(order), len(encoded))]
        micro = 0
        for epoch in epochs:
            model.train()
            kd_lora_tree.new_epoch_init(len(epoch))
            for index in epoch:
                if config.reg > 0:
                    kd_lora_tree.step()
                ids, labels = encoded[index]
                loss = model(input_ids=torch.tensor([ids]), labels=torch.tensor([labels]), use_cache=False).loss
                if config.reg > 0:
                    _grad_current = []
                    for name_, param_ in model.named_parameters():
                        if "lora_A" in name_:
                            _grad_current.append(param_)
                    width = max(p.numel() for p in _grad_current)
                    _grad_current = torch.stack([torch.nn.functional.pad(p.reshape(-1), (0, width - p.numel()))
                                                 for p in _grad_current], dim=0)
                    kd_lora_tree.insert_grad(_grad_current)
                    if task_id > 0:
                        prev_id_matrix = kd_lora_tree.tree_search(task_id, device=torch.device("cpu"))
                        reg_loss = kd_lora_tree.get_loss(_grad_current, loss, task_id, prev_id_matrix)
                        loss = loss - reg_loss
                (loss / batch_size).backward()
                micro += 1
                if micro % batch_size == 0:
                    torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        kd_lora_tree.end_task(task_id=task_id)
        fingerprints.append(kd_lora_tree.all_accumulate_grads[task_id].clone())
    return fingerprints


def local_stream(model, tasks, *, config, batch_size, updates, seed, interrupt_at=None):
    """The baseline as run.py drives it: one job per stage, the tree rebuilt from sealed fingerprints and the
    optimizer state carried; optionally one stage interrupted after `interrupt_at` updates and resumed."""
    import torch

    from run import restore_rng_state, rng_state

    fingerprints, optimizer_state, logs = [], None, []
    for task_id, records in enumerate(tasks):
        tree = new_tree(len(tasks), config.reg, [fp.clone() for fp in fingerprints])
        optimizer = make_optimizer(model, config)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        selection = torch.zeros((task_id, stacked_lora_a(model).shape[0]), dtype=torch.long) if task_id else None
        saved = {}

        def checkpoint(progress):
            saved.update(progress=progress, lora=lora_state(model), optimizer=copy.deepcopy(optimizer.state_dict()),
                         tree=copy.deepcopy(tree_epoch_state(tree)), rng=rng_state(),
                         selection=None if selection is None else selection.clone())

        kwargs = dict(task_id=task_id, batch_size=batch_size, config=config, seed=seed + task_id,
                      max_length=10_000, log=lambda row: logs.append((task_id, row)))
        if interrupt_at is not None and task_id == len(tasks) - 1:
            train_stage(model, FakeTokenizer(), optimizer, tree, records, updates=updates, selection=selection,
                        checkpoint_every=interrupt_at, checkpoint=checkpoint, **kwargs)
            # Resume from the checkpoint in a fresh "job": reload weights, optimizer, tree and RNG.
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in saved["lora"]:
                        parameter.copy_(saved["lora"][name])
            tree = new_tree(len(tasks), config.reg, [fp.clone() for fp in fingerprints])
            restore_tree_epoch_state(tree, saved["tree"], task_id, torch.device("cpu"))
            optimizer = make_optimizer(model, config)
            optimizer.load_state_dict(saved["optimizer"])
            restore_rng_state(saved["rng"])
            selection = saved["selection"]
            train_stage(model, FakeTokenizer(), optimizer, tree, records, updates=updates, selection=selection,
                        resume=saved["progress"], **kwargs)
        else:
            train_stage(model, FakeTokenizer(), optimizer, tree, records, updates=updates, selection=selection,
                        **kwargs)
        tree.end_task(task_id=task_id)
        fingerprints.append(tree.all_accumulate_grads[task_id].clone())
        optimizer_state = copy.deepcopy(optimizer.state_dict())
    return fingerprints, logs


@pytest.mark.parametrize("interrupt_at", [None, 1])
def test_local_stage_loop_matches_a_literal_upstream_transcription(interrupt_at):
    import torch

    config = TreeLoraConfig()
    tasks = [RECORDS, [dict(r, answer=r["answer"][::-1]) for r in RECORDS], RECORDS[::-1]]
    model = tiny_lora_model()
    reference, local = copy.deepcopy(model), copy.deepcopy(model)
    torch.manual_seed(7)
    expected = reference_stream(reference, tasks, config=config, batch_size=4, updates=3, seed=11)
    torch.manual_seed(7)
    actual, logs = local_stream(local, tasks, config=config, batch_size=4, updates=3, seed=11,
                                interrupt_at=interrupt_at)
    assert all(row["regLoss"] == 0 for task_id, row in logs if task_id == 0)
    assert all(row["regLoss"] != 0 for task_id, row in logs if task_id > 0)
    for want, got in zip(expected, actual, strict=True):
        assert torch.allclose(want, got, atol=1e-6, rtol=1e-5)
    want_state, got_state = lora_state(reference), lora_state(local)
    assert want_state.keys() == got_state.keys()
    for name in want_state:
        assert torch.allclose(want_state[name], got_state[name], atol=1e-6, rtol=1e-5), name
