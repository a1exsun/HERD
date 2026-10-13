from __future__ import annotations


def text_backbone_lora_targets(model) -> list[str]:
    """Every text-backbone Linear module, excluding the vision tower, MTP head and LM head."""
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
