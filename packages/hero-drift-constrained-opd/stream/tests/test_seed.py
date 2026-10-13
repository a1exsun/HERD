"""Real PEFT initialization on a tiny CPU linear model; no downloaded checkpoint."""
from copy import deepcopy
from pathlib import Path
import sys

import torch
from peft import LoraConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hero


def test_adapter_seed_is_stage_local_and_controls_initialization():
    base = torch.nn.Sequential(torch.nn.Linear(8, 8, bias=False))

    def initialize(seed, stage, ambient):
        torch.manual_seed(ambient)
        torch.randn(103)  # arbitrary work before restarting a stage
        model = hero.initialize_adapter(deepcopy(base), LoraConfig(r=2, target_modules=['0']),
                                        seed=seed, stage=stage)
        return {n: p.detach().clone() for n, p in model.named_parameters() if 'lora_' in n}

    first = initialize(42, 7, 111)
    resumed = initialize(42, 7, 999)
    assert all(torch.equal(first[n], resumed[n]) for n in first)
    for changed in (initialize(43, 7, 111), initialize(42, 8, 111)):
        assert any(not torch.equal(first[n], changed[n]) for n in first if 'lora_A' in n)
    assert all(torch.count_nonzero(v) == 0 for n, v in first.items() if 'lora_B' in n)
