"""Calibration metrics (design doc section 4, step 3).

These are the numbers that decide whether `confidence` is shippable. Calibration is a
POPULATION property: "0.9 means right about 90% of the time across many predictions"
says nothing about any individual answer. Report these, not vibes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Bin:
    lo: float
    hi: float
    count: int
    confidence: float
    accuracy: float


@dataclass(frozen=True)
class SelectivePoint:
    threshold: float
    coverage: float
    accuracy: float | None


def _check(confidences: Sequence[float], correct: Sequence[bool]) -> None:
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    if not confidences:
        raise ValueError("cannot measure calibration on an empty sample")


def reliability_bins(
    confidences: Sequence[float], correct: Sequence[bool], bins: int = 10
) -> list[Bin]:
    """Equal-width bins over [0,1]. The raw material for a reliability diagram."""
    _check(confidences, correct)
    width = 1.0 / bins
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for c, ok in zip(confidences, correct):
        index = min(int(c / width), bins - 1)  # c == 1.0 belongs in the top bin
        buckets[index].append((c, ok))

    rows = []
    for i, bucket in enumerate(buckets):
        n = len(bucket)
        rows.append(
            Bin(
                lo=i * width,
                hi=(i + 1) * width,
                count=n,
                confidence=sum(c for c, _ in bucket) / n if n else 0.0,
                accuracy=sum(ok for _, ok in bucket) / n if n else 0.0,
            )
        )
    return rows


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], bins: int = 10
) -> float:
    """Population-weighted mean gap between confidence and accuracy."""
    _check(confidences, correct)
    total = len(confidences)
    return sum(
        (b.count / total) * abs(b.accuracy - b.confidence)
        for b in reliability_bins(confidences, correct, bins)
        if b.count
    )


def brier_score(
    probabilities: Sequence[Mapping[str, float]], labels: Sequence[str]
) -> float:
    """Multiclass Brier: mean sum_k (p_k - y_k)^2. A proper scoring rule, so it
    rewards honest probabilities rather than confident guesses."""
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must be the same length")
    if not probabilities:
        raise ValueError("cannot score an empty sample")

    total = 0.0
    for dist, label in zip(probabilities, labels):
        if label not in dist:
            raise ValueError(f"label {label!r} is not among the options {sorted(dist)}")
        total += sum((p - (1.0 if k == label else 0.0)) ** 2 for k, p in dist.items())
    return total / len(probabilities)


def selective_accuracy(
    confidences: Sequence[float],
    correct: Sequence[bool],
    thresholds: Sequence[float],
) -> list[SelectivePoint]:
    """Accuracy among predictions that clear each confidence bar, with coverage.

    This is the curve a confidence-gated router actually rides: it answers "if I
    escalate everything below t, how much do I still handle and how often am I right?"
    """
    _check(confidences, correct)
    total = len(confidences)
    rows = []
    for t in thresholds:
        kept = [ok for c, ok in zip(confidences, correct) if c >= t]
        rows.append(
            SelectivePoint(
                threshold=t,
                coverage=len(kept) / total,
                accuracy=(sum(kept) / len(kept)) if kept else None,
            )
        )
    return rows
