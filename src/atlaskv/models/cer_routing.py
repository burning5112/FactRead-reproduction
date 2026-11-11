"""CER routing and hierarchical support-loss helpers."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def entropy_aware_candidate_scores(logits: Tensor, temperature: float) -> Tensor:
    """Aggregate candidate probabilities across each query-token/head unit."""
    if logits.ndim != 4 or any(size == 0 for size in logits.shape):
        raise ValueError("logits must have non-empty (batch, heads, query, candidates) dimensions")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")

    probabilities = torch.softmax(logits.float(), dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    weights = torch.exp(-entropy / temperature)
    weighted_mass = (probabilities * weights.unsqueeze(-1)).sum(dim=(1, 2))
    normalizer = weights.sum(dim=(1, 2)).unsqueeze(-1).clamp_min(1e-12)
    return weighted_mass / normalizer


def hierarchical_support_loss_from_logits(
    logits: Tensor,
    candidate_ids: Tensor,
    support_mask: Tensor,
    temperature: float = 0.1,
    epsilon: float = 1e-12,
) -> Tensor:
    """Compute HSP after merging duplicate candidate node IDs per example."""
    if logits.ndim != 4 or logits.shape[-1] == 0:
        raise ValueError("logits must have shape (batch, heads, tokens, candidates)")
    expected_shape = (logits.shape[0], logits.shape[-1])
    if tuple(candidate_ids.shape) != expected_shape or tuple(support_mask.shape) != expected_shape:
        raise ValueError("candidate_ids and support_mask must match batch and candidate dimensions")
    if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("temperature and epsilon must be finite and positive")

    candidate_ids = candidate_ids.to(device=logits.device, dtype=torch.long)
    support_mask = support_mask.to(device=logits.device, dtype=torch.bool)
    if not torch.all(support_mask.any(dim=-1)):
        raise ValueError("every example must contain at least one supported node")

    example_losses = []
    for row in range(logits.shape[0]):
        _, inverse = torch.unique(candidate_ids[row], sorted=True, return_inverse=True)
        unique_count = int(inverse.max().item()) + 1
        multiplicity = torch.bincount(inverse, minlength=unique_count).to(dtype=torch.float32)
        adjusted_logits = logits[row].float() - multiplicity[inverse].log()[None, None, :]
        row_evidence = torch.softmax(adjusted_logits, dim=-1).mean(dim=(0, 1))
        unique_evidence = row_evidence.new_zeros(unique_count).scatter_add_(0, inverse, row_evidence)
        support_count = torch.zeros(unique_count, dtype=torch.long, device=logits.device).scatter_add_(
            0, inverse, support_mask[row].long()
        )
        if torch.any((support_count != 0) & (support_count != multiplicity.long())).item():
            raise ValueError("duplicate candidate IDs must have consistent support labels")
        unique_support = support_count > 0

        sharpened = torch.softmax(torch.log(unique_evidence + epsilon) / temperature, dim=-1)
        supported_mass = sharpened.masked_fill(~unique_support, 0.0).sum()
        example_losses.append(-torch.log(supported_mass.clamp_min(epsilon)))
    return torch.stack(example_losses).mean()


def hierarchical_support_loss_from_levels(
    level_logits: tuple[Tensor, ...],
    candidate_ids: tuple[Tensor, ...],
    support_masks: tuple[Tensor, ...] | None,
    *,
    weight: float,
    temperature: float,
) -> Tensor | None:
    """Return the mean level loss when enabled; zero weight is a true no-op."""
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("weight must be finite and non-negative")
    if weight == 0:
        return None
    if support_masks is None:
        raise ValueError("CER HSP requires root/inter/leaf support masks")
    if not level_logits or not (len(level_logits) == len(candidate_ids) == len(support_masks)):
        raise ValueError("level logits, candidate IDs, and support masks must align")

    losses = [
        hierarchical_support_loss_from_logits(
            logits,
            ids,
            mask,
            temperature=temperature,
        )
        for logits, ids, mask in zip(level_logits, candidate_ids, support_masks)
    ]
    return torch.stack(losses).mean()
