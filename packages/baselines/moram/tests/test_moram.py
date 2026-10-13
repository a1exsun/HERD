import pytest
import torch
import torch.nn.functional as F

from moram_cl import (
    MoramConfig, atom_counts, batch_schedule, increment_all, inject_moram, load_moram_state, make_optimizer,
    moram_modules, moram_state, moram_targets, supervised_ids,
)
from upstream.moram import MoRAMLinear


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8, bias=False)
        self.down_proj = torch.nn.Linear(8, 8, bias=True)

    def forward(self, x):
        return self.down_proj(torch.relu(self.q_proj(x)))


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Block(), Block()])
        self.visual = torch.nn.Linear(8, 8)
        self.lm_head = torch.nn.Linear(8, 5, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


CONFIG = MoramConfig(rank=4, topk=4, lora_alpha=8, lora_dropout=0.0)


def injected(seed=0):
    torch.manual_seed(seed)
    model = Tiny()
    originals = {name: module.weight for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    inject_moram(model, CONFIG, moram_targets(model))
    return model, originals


def test_targets_exclude_vision_and_lm_head_and_share_the_frozen_base_weights():
    model, originals = injected()
    names = sorted(name for name, _ in moram_modules(model))
    assert names == ["layers.0.down_proj", "layers.0.q_proj", "layers.1.down_proj", "layers.1.q_proj"]
    for name, module in moram_modules(model):
        assert module.weight is originals[name] and not module.weight.requires_grad
        assert module.router_temperature == CONFIG.router_temperature
        assert module.moram_infer_lora_a_thresh == CONFIG.infer_threshold
    trainable = sorted(name for name, p in model.named_parameters() if p.requires_grad)
    assert trainable and all("loranew_" in name for name in trainable)


def test_a_fresh_bank_leaves_the_base_function_unchanged():
    model, _ = injected()
    reference = Tiny()
    reference.load_state_dict({k: v for k, v in model.state_dict().items() if "lora" not in k})
    x = torch.randn(3, 8)
    assert torch.allclose(model(x), reference(x), atol=1e-6)  # new atoms start with B = 0


def test_forward_matches_the_documented_self_activation_rule():
    torch.manual_seed(1)
    layer = MoRAMLinear("default", 6, 5, r=3, lora_alpha=6, lora_dropout=0.0, r_sum=0, moram_topk=2)
    layer.set_router_temperature(0.5)
    with torch.no_grad():
        layer.loranew_B["default"].weight.normal_()
    x = torch.randn(4, 6)
    projection = x @ layer.loranew_A["default"].weight.T
    normed = F.normalize(projection, dim=-1)
    top = torch.topk(normed, 2, dim=-1)
    logits = torch.full_like(normed, float("-inf")).scatter(-1, top.indices, top.values)
    weights = torch.softmax(logits / 0.5, dim=-1)
    expected = x @ layer.weight.T + layer.bias + (projection * weights) @ (layer.loranew_B["default"].weight * 2).T
    layer.train()
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_increment_appends_trained_atoms_frozen_and_reinitializes_new_ones():
    model, _ = injected()
    with torch.no_grad():
        for _, module in moram_modules(model):
            module.loranew_B["default"].weight.fill_(0.5)
    trained = {name: (module.loranew_A["default"].weight.clone(), module.loranew_B["default"].weight.clone())
               for name, module in moram_modules(model)}
    increment_all(model, seed=3)
    for name, module in moram_modules(model):
        assert torch.equal(module.lora_A["default"].weight, trained[name][0])
        # Upstream stores the unscaled B of the finished task in the frozen bank.
        assert torch.equal(module.lora_B["default"].weight, trained[name][1])
        assert not module.lora_A["default"].weight.requires_grad
        assert torch.count_nonzero(module.loranew_B["default"].weight) == 0
        assert module.loranew_A["default"].weight.requires_grad
    assert atom_counts(model)["consolidatedAtomsPerModule"] == CONFIG.rank
    increment_all(model, seed=4)
    assert atom_counts(model)["consolidatedAtomsPerModule"] == 2 * CONFIG.rank


def test_increment_is_reproducible_and_state_round_trips_into_a_fresh_model():
    first, _ = injected()
    increment_all(first, seed=9)
    second, _ = injected()
    increment_all(second, seed=9)
    state = moram_state(first)
    assert all(torch.equal(state[key], value) for key, value in moram_state(second).items())
    restored, _ = injected(seed=5)
    restored.load_state_dict({k: v for k, v in first.state_dict().items() if "lora" not in k}, strict=False)
    load_moram_state(restored, state)
    assert all(torch.equal(state[key], value) for key, value in moram_state(restored).items())
    x = torch.randn(2, 8)
    first.eval(), restored.eval()
    assert torch.allclose(first(x), restored(x), atol=1e-6)
    with pytest.raises(ValueError, match="does not match"):
        load_moram_state(restored, {k: v for k, v in state.items() if not k.startswith("layers.0.q_proj")})


def test_inference_gate_applies_only_in_eval_mode():
    torch.manual_seed(2)
    layer = MoRAMLinear("default", 6, 5, r=4, lora_alpha=8, lora_dropout=0.0, r_sum=0, moram_topk=4)
    layer.set_router_temperature(1.0)
    layer.set_moram_infer_lora_a_threshold(0.99)  # nearly every atom falls below the gate
    with torch.no_grad():
        layer.loranew_B["default"].weight.normal_()
    x = torch.randn(3, 6)
    base = x @ layer.weight.T + layer.bias
    layer.train()
    training = layer(x)
    layer.eval()
    gated = layer(x)
    assert not torch.allclose(training, base, atol=1e-4)
    assert (gated - base).abs().max() < (training - base).abs().max()


def test_optimizer_state_carries_across_stages_over_the_same_parameters():
    model, _ = injected()
    optimizer = make_optimizer(model, CONFIG)
    loss = model(torch.randn(2, 8)).pow(2).sum()
    loss.backward()
    optimizer.step()
    increment_all(model, seed=1)
    carried = make_optimizer(model, CONFIG)
    carried.load_state_dict(optimizer.state_dict())
    assert carried.param_groups[0]["betas"] == (0.9, 0.95)
    assert len(carried.state) == len(optimizer.state)


class FakeTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        ids = [1, 2, 3] if add_generation_prompt else [1, 2, 3] + [ord(c) for c in messages[-1]["content"]]
        return {"input_ids": ids} if kwargs.get("return_dict") else ids


def test_supervised_targets_and_schedule():
    ids, labels = supervised_ids(FakeTokenizer(), {"id": "x", "prompt": "p", "references": ["ok"]}, max_length=8)
    assert labels == [-100, -100, -100, ord("o"), ord("k")]
    with pytest.raises(ValueError, match="exceeds"):
        supervised_ids(FakeTokenizer(), {"id": "x", "prompt": "p", "answer": "ok"}, max_length=4)
    schedule = batch_schedule(3, updates=2, batch_size=3, seed=5)
    assert schedule == batch_schedule(3, updates=2, batch_size=3, seed=5)
    assert all(sorted(batch) == [0, 1, 2] for batch in schedule)
