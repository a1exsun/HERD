"""Padded micro-batches for the per-record supervised objective.

The objective is the per-record one: each record contributes its token-mean cross-entropy divided by the
batch size. Records of one update are right-padded into micro-batches under a padded-token budget, so every
valid position sees exactly the context it sees alone (the model is causal and padding follows the
sequence). Only answer positions are projected to the vocabulary, in recomputed chunks.
"""
from __future__ import annotations

from typing import Any, Sequence

TOKEN_BUDGET = 24576  # padded tokens per micro-batch
LOGIT_CHUNK = 2048  # answer positions projected to the vocabulary at once


def backbone_and_head(model):
    """The decoder stack (returning final-norm hidden states) and the LM head of a plain or PEFT model."""
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    return inner.model, inner.lm_head


def micro_batches(lengths: Sequence[int], budget: int = TOKEN_BUDGET) -> list[list[int]]:
    """Positions of `lengths` grouped longest first so that rows x longest row stays within the budget."""
    order = sorted(range(len(lengths)), key=lambda k: (-lengths[k], k))
    groups: list[list[int]] = []
    current: list[int] = []
    width = 0
    for k in order:
        if current and max(width, lengths[k]) * (len(current) + 1) > budget:
            groups.append(current)
            current, width = [], 0
        current.append(k)
        width = max(width, lengths[k])
    if current:
        groups.append(current)
    return groups


def backward_records(model, encoded: Sequence[tuple[list[int], list[int]]], *, batch_size: int, pad_id: int,
                     budget: int = TOKEN_BUDGET) -> list[Any]:
    """Backpropagate sum_r CE_r / batch_size over `encoded` and return each record's token-mean
    cross-entropy as a detached scalar tensor, in input order."""
    import torch
    from torch.utils.checkpoint import checkpoint

    backbone, head = backbone_and_head(model)
    device = head.weight.device

    def chunk_loss(hidden, targets):
        return torch.nn.functional.cross_entropy(head(hidden).float(), targets, reduction="none")

    losses: list[Any] = [None] * len(encoded)
    for group in micro_batches([len(ids) for ids, _ in encoded], budget):
        width = max(len(encoded[k][0]) for k in group)
        input_ids = torch.full((len(group), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(group), width), dtype=torch.long)
        rows, cols, targets, weights, spans = [], [], [], [], []
        for row, k in enumerate(group):
            ids, labels = encoded[k]
            input_ids[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            mask[row, :len(ids)] = 1
            positions = [t for t in range(1, len(labels)) if labels[t] != -100]
            spans.append((len(targets), len(targets) + len(positions)))
            rows += [row] * len(positions)
            cols += [t - 1 for t in positions]
            targets += [labels[t] for t in positions]
            weights += [1.0 / (len(positions) * batch_size)] * len(positions)
        hidden = backbone(input_ids=input_ids.to(device), attention_mask=mask.to(device),
                          use_cache=False).last_hidden_state
        picked = hidden[torch.tensor(rows, device=device), torch.tensor(cols, device=device)]
        target = torch.tensor(targets, dtype=torch.long, device=device)
        token_loss = torch.cat([checkpoint(chunk_loss, picked[start:start + LOGIT_CHUNK],
                                           target[start:start + LOGIT_CHUNK], use_reentrant=False)
                                for start in range(0, len(targets), LOGIT_CHUNK)])
        (token_loss * torch.tensor(weights, device=device)).sum().backward()
        detached = token_loss.detach()
        for row, k in enumerate(group):
            start, end = spans[row]
            losses[k] = detached[start:end].mean()
        del hidden, picked, token_loss
    return losses
