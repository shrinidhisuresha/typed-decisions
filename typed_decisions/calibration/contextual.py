"""Contextual calibration (design doc section 4, step 1).

Ask the same question with a content-free state to see what the model would have said
anyway, then divide that prior out. This kills the model's standing preference for
particular label tokens -- a bias that has nothing to do with the state and would
otherwise be baked into every answer.

Unsupervised: it needs one extra forward pass, not labelled data, so it can run in
Phase 1 before you have outcomes. After Zhao et al., "Calibrate Before Use".
"""

from __future__ import annotations

from typing import Sequence

NULL_STATE = "N/A"

_EPSILON = 1e-12


def debias(probabilities: Sequence[float], null_prior: Sequence[float]) -> list[float]:
    """Divide out the null-state prior and renormalise.

    If the answer exactly matches the prior, the result is uniform -- correctly saying
    the state added nothing.
    """
    if len(probabilities) != len(null_prior):
        raise ValueError("probabilities and null_prior must be the same length")
    if sum(null_prior) <= 0.0:
        raise ValueError("null prior must contain some mass")

    adjusted = [p / max(q, _EPSILON) for p, q in zip(probabilities, null_prior)]
    total = sum(adjusted)
    if total <= 0.0:
        n = len(adjusted)
        return [1.0 / n] * n
    return [a / total for a in adjusted]
