from hero_baselines.controls import LoraControlConfig


def test_lora_control_matches_the_shared_parametric_budget():
    lora = LoraControlConfig()
    assert (lora.rank, lora.alpha, lora.dropout) == (32, 64, 0.0)
    assert (lora.learning_rate, lora.fixed_updates, lora.effective_batch_size, lora.seed) == (1e-4, 32, 32, 42)
