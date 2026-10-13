"""Lossless EMA persistence, including frozen-base identity and current LoRA deltas."""
import copy

import pytest
import torch

from sdft import EmaTeacher


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_checkpoint_preserves_parameter_bytes_and_rng(tmp_path, dtype):
    base = torch.nn.Module()
    base.weight = torch.nn.Parameter(torch.tensor([0.0, -0.0, 1.0, -1.0, float('inf'), float('-inf'), float('nan')], dtype=dtype))
    base.scalar = torch.nn.Parameter(torch.tensor(0.5, dtype=dtype))
    teacher = EmaTeacher(copy.deepcopy(base), base, 0.01)
    with torch.no_grad():
        teacher.model.weight[2] = 1.125
        teacher.model.scalar.fill_(-0.0)
    expected = {n: p.detach().reshape(-1).view(torch.uint8).clone() for n, p in teacher.model.named_parameters()}
    rng = torch.get_rng_state().clone()
    checkpoint = tmp_path / 'teacher.xor.gz'
    teacher.save_checkpoint(checkpoint)
    restored = EmaTeacher(copy.deepcopy(base), base, 0.01)
    restored.load_checkpoint(checkpoint)
    assert torch.equal(rng, torch.get_rng_state())
    for name, parameter in restored.model.named_parameters():
        assert torch.equal(parameter.detach().reshape(-1).view(torch.uint8), expected[name])
    with torch.no_grad():
        base.scalar.fill_(1)
    with pytest.raises(ValueError, match='base does not match'):
        restored.load_checkpoint(checkpoint)


def test_checkpoint_uses_frozen_peft_base_and_retains_exact_ema(tmp_path):
    from peft import LoraConfig, get_peft_model

    base = torch.nn.Sequential(torch.nn.Linear(4, 3, bias=False))
    original = copy.deepcopy(base)
    student = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=['0']))
    teacher = EmaTeacher(copy.deepcopy(original), student, 0.25)
    with torch.no_grad():
        student.get_base_model()[0].lora_B['default'].weight.fill_(0.5)
    teacher.update()
    expected = teacher.model[0].weight.detach().clone()
    checkpoint = tmp_path / 'teacher.xor.gz'
    teacher.save_checkpoint(checkpoint)
    # A new adapter delta must not change the frozen reconstruction anchor.
    with torch.no_grad():
        student.get_base_model()[0].lora_B['default'].weight.fill_(1.5)
    restored = EmaTeacher(copy.deepcopy(original), student, 0.25)
    restored.load_checkpoint(checkpoint)
    assert torch.equal(restored.model[0].weight, expected)


def test_unchanged_weights_are_compact_and_corruption_is_rejected(tmp_path):
    base = torch.nn.Linear(1024, 256, bias=False)
    teacher = EmaTeacher(copy.deepcopy(base), base, 0.01)
    checkpoint = tmp_path / 'teacher.xor.gz'
    teacher.save_checkpoint(checkpoint)
    assert checkpoint.stat().st_size < base.weight.numel() * base.weight.element_size() // 100
    checkpoint.write_bytes(checkpoint.read_bytes()[:-12])
    with pytest.raises((EOFError, ValueError)):
        teacher.load_checkpoint(checkpoint)
