"""The training objective, on the same distribution scoring.py serves."""

from __future__ import annotations

import torch


def restricted_log_probs(last_logits: torch.Tensor, label_ids: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
    """Log-softmax over each row's label tokens only.

    last_logits: (B, V) logits at the `Answer:` position. label_ids: (B, K) token ids,
    padded. mask: (B, K) bool, False on padding. Padding gets -inf, so it carries no mass.
    """
    picked = last_logits.float().gather(1, label_ids)
    picked = picked.masked_fill(~mask, float("-inf"))
    return torch.log_softmax(picked, dim=-1)


def log_score(logp: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against a (possibly soft) target: the log score, a proper scoring rule.
    Minimised only by the true conditional distribution, which is what makes the result
    calibrated rather than merely accurate."""
    safe = logp.masked_fill(~mask, 0.0)
    return -(target * safe).sum(dim=-1)


def ranked_probability_score(probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """RPS for ordinal answers (Score): squared error between cumulative distributions, so
    predicting "medium" for a "high" costs less than predicting "low". Proper as well."""
    return ((probs.cumsum(-1) - target.cumsum(-1)) ** 2).sum(dim=-1) / max(probs.shape[-1] - 1, 1)


def ordinal_rps(logp: torch.Tensor, target: torch.Tensor, order: list[tuple[int, ...]]) -> torch.Tensor:
    """RPS per row, over the levels re-sorted into their natural order (the prompt may show
    them shuffled). Rows with no ordinal order contribute 0."""
    out = []
    for i, idx in enumerate(order):
        if not idx:
            out.append(logp.new_zeros(()))
            continue
        ix = torch.tensor(idx, device=logp.device)
        out.append(ranked_probability_score(logp[i, ix].exp()[None], target[i, ix][None])[0])
    return torch.stack(out)
