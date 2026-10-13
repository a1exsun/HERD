"""Recovery from a sealed frozen bank plus current-task atoms and Adam/RNG state."""
import copy

import pytest
import torch

from moram_cl import (MoramConfig, increment_all, inject_moram, load_moram_state, load_trainable_atom_state,
                      make_optimizer, moram_state, moram_targets, trainable_atom_state)


def step(model, optimizer, x):
    optimizer.zero_grad()
    model(x).square().sum().backward()
    optimizer.step()


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_resume_matches_full_state_and_next_adam_step_with_dropout(dtype):
    torch.manual_seed(7)
    base = torch.nn.Sequential(torch.nn.Linear(8, 8, dtype=dtype))
    config = MoramConfig(rank=2, topk=2, lora_alpha=4, lora_dropout=0.1)
    model = copy.deepcopy(base)
    inject_moram(model, config, moram_targets(model))
    optimizer = make_optimizer(model, config)
    x = torch.randn(5, 8, dtype=dtype)
    step(model, optimizer, x)
    increment_all(model, seed=9)
    previous = {k: v.clone() for k, v in moram_state(model).items()}
    optimizer = make_optimizer(model, config)
    step(model, optimizer, x)
    checkpoint = {k: v.clone() for k, v in trainable_atom_state(model).items()}
    adam = copy.deepcopy(optimizer.state_dict())
    rng = torch.get_rng_state().clone()
    full = {k: v.clone() for k, v in moram_state(model).items()}
    assert len(checkpoint) * 2 == len(full)
    assert all(torch.equal(full[k], previous[k]) for k in full if k not in checkpoint)
    step(model, optimizer, x)
    expected = {k: v.clone() for k, v in moram_state(model).items()}

    restored = copy.deepcopy(base)
    inject_moram(restored, config, moram_targets(restored))
    load_moram_state(restored, previous)
    load_trainable_atom_state(restored, checkpoint)
    assert all(torch.equal(v, moram_state(restored)[k]) for k, v in full.items())
    restored_optimizer = make_optimizer(restored, config)
    restored_optimizer.load_state_dict(adam)
    torch.set_rng_state(rng)
    step(restored, restored_optimizer, x)
    assert all(torch.equal(v, moram_state(restored)[k]) for k, v in expected.items())


@pytest.mark.parametrize('corruption', ['name', 'shape', 'dtype'])
def test_invalid_checkpoint_is_rejected_before_any_atom_changes(corruption):
    model = torch.nn.Sequential(torch.nn.Linear(4, 3))
    inject_moram(model, MoramConfig(rank=2, topk=2), moram_targets(model))
    state = {k: v.clone() for k, v in trainable_atom_state(model).items()}
    before = {k: v.clone() for k, v in state.items()}
    key = next(iter(state))
    if corruption == 'name':
        state.pop(key)
    elif corruption == 'shape':
        state[key] = state[key][:1]
    else:
        state[key] = state[key].to(torch.float64)
    with pytest.raises(ValueError, match='checkpoint'):
        load_trainable_atom_state(model, state)
    assert all(torch.equal(v, trainable_atom_state(model)[k]) for k, v in before.items())
