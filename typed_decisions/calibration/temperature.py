"""Temperature scaling (design doc section 4, step 2).

One parameter, fitted by NLL on held-out labelled data. It usually does most of the
work, so it is the first thing to reach for -- and unlike the debiasing steps it needs
labels, which is why it comes after contextual calibration in the pipeline.
"""

from __future__ import annotations

import math
from typing import Sequence

LogitRows = Sequence[Sequence[float]]


def _check(rows: LogitRows, labels: Sequence[int]) -> None:
    if len(rows) != len(labels):
        raise ValueError("logit rows and labels must be the same length")
    if not rows:
        raise ValueError("cannot fit a temperature on an empty sample")
    for row, label in zip(rows, labels):
        if not 0 <= label < len(row):
            raise ValueError(f"label {label} is out of range for {len(row)} options")


def negative_log_likelihood(rows: LogitRows, labels: Sequence[int], temperature: float) -> float:
    """Mean NLL of the true option under softmax(logits / T)."""
    total = 0.0
    for row, label in zip(rows, labels):
        scaled = [z / temperature for z in row]
        peak = max(scaled)
        logsumexp = peak + math.log(sum(math.exp(z - peak) for z in scaled))
        total += logsumexp - scaled[label]
    return total / len(rows)


def fit_temperature(
    rows: LogitRows,
    labels: Sequence[int],
    bounds: tuple[float, float] = (0.001, 20.0),
    tolerance: float = 1e-4,
) -> float:
    """Golden-section search over log T. NLL is unimodal in T, so this is enough --
    and it keeps the package free of a scipy dependency."""
    _check(rows, labels)

    lo, hi = math.log(bounds[0]), math.log(bounds[1])
    invphi = (math.sqrt(5.0) - 1.0) / 2.0

    a, b = lo, hi
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc = negative_log_likelihood(rows, labels, math.exp(c))
    fd = negative_log_likelihood(rows, labels, math.exp(d))

    while abs(b - a) > tolerance:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = negative_log_likelihood(rows, labels, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = negative_log_likelihood(rows, labels, math.exp(d))

    return math.exp((a + b) / 2.0)


# Zero probabilities must be treated the SAME way when a temperature is fitted and when it
# is applied. An earlier version floored zeros at 1e-12 in the fit (about 0.25 relative mass
# at T=20) while sharpen() kept them at exactly 0, so the fit optimised a distribution that
# was never produced. Inputs with many exact zeros -- rounded probabilities, or options cut
# by retrieval -- pushed the fit to its bound. An independent audit caught it.
DEFAULT_FLOOR = 1e-6


def floored(probabilities: Sequence[float], floor: float = DEFAULT_FLOOR) -> list[float]:
    """max(p, floor), renormalised. The one place zeros are made finite."""
    if floor <= 0.0:
        return list(probabilities)
    raised = [max(p, floor) for p in probabilities]
    total = sum(raised)
    return [p / total for p in raised]


def sharpen(probabilities: Sequence[float], temperature: float,
            floor: float = DEFAULT_FLOOR) -> list[float]:
    """Apply a fitted temperature in probability space: p^(1/T), renormalised.

    Equivalent to softmax(log p / T), which is what lets the fitted temperature sit at
    the END of the pipeline -- after permutation averaging and contextual debiasing --
    rather than being tangled up with the raw logits. T=1 is the identity. Otherwise the
    input is floored exactly as the fit floored it (see DEFAULT_FLOOR).
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if temperature == 1.0:
        return list(probabilities)
    base = floored(probabilities, floor)
    powered = [p ** (1.0 / temperature) if p > 0.0 else 0.0 for p in base]
    total = sum(powered)
    if total <= 0.0:
        n = len(powered)
        return [1.0 / n] * n
    return [p / total for p in powered]
