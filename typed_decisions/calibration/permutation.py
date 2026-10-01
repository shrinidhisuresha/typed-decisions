"""Option-order debiasing (design doc section 4, step 1).

LLMs have a strong, well-documented positional bias towards the first (or last) option.
Scoring the same question under several orderings and averaging cancels it.

Cyclic rotations rather than random permutations: deterministic, reproducible, and with
k == n every option occupies every position exactly once, which cancels a pure
positional bias exactly rather than approximately.

Costs k forward passes per question -- but they all fork off the same cached state,
so the marginal cost is the branch, not the prefill.
"""

from __future__ import annotations

from typing import Sequence

Ordering = tuple[int, ...]


def rotations(n: int, k: int) -> list[Ordering]:
    """k orderings of n options. ordering[i] is the ORIGINAL index shown at position i."""
    if n < 1:
        raise ValueError("need at least one option to permute")
    count = max(1, min(k, n))
    return [tuple((i + r) % n for i in range(n)) for r in range(count)]


def unpermute(probabilities: Sequence[float], ordering: Ordering) -> list[float]:
    """Send each position's probability back to the option that actually sat there."""
    if len(probabilities) != len(ordering):
        raise ValueError("probabilities and ordering must be the same length")
    out = [0.0] * len(ordering)
    for position, original in enumerate(ordering):
        out[original] = probabilities[position]
    return out


def average_distributions(distributions: Sequence[Sequence[float]]) -> list[float]:
    """Mean of several distributions over the same option set, renormalised."""
    if not distributions:
        raise ValueError("cannot average an empty list of distributions")
    n = len(distributions[0])
    if any(len(d) != n for d in distributions):
        raise ValueError("all distributions must cover the same options")

    summed = [sum(d[i] for d in distributions) for i in range(n)]
    total = sum(summed)
    if total <= 0.0:
        return [1.0 / n] * n
    return [s / total for s in summed]
