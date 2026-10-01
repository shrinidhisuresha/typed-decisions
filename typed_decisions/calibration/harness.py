"""The Phase 1 harness: measure calibration, fit a temperature, measure again.

Do not ship a confidence number until `report()` says the gap is small on YOUR data.
An uncalibrated confidence is worse than none, because callers will gate on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from ..scoring import confidence_from
from .metrics import (
    Bin,
    SelectivePoint,
    brier_score,
    expected_calibration_error,
    reliability_bins,
    selective_accuracy,
)
from .temperature import (
    DEFAULT_FLOOR,
    fit_temperature,
    floored,
    negative_log_likelihood,
    sharpen,
)

DEFAULT_THRESHOLDS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


@dataclass(frozen=True)
class Prediction:
    """One scored example: the final distribution, and what was actually true."""

    probabilities: Mapping[str, float]
    label: str


@dataclass(frozen=True)
class TemperatureFit:
    """A fitted temperature plus enough context to tell whether to trust it."""

    temperature: float
    at_bound: bool
    nll_before: float
    nll_after: float
    # Every example already ranked correctly: NLL keeps falling as T -> 0, so the "fit"
    # is just wherever the search stopped. It carries no information about T.
    separable: bool = False

    @property
    def trustworthy(self) -> bool:
        return not (self.at_bound or self.separable)

    def summary(self) -> str:
        flag = ("  [SEPARABLE - no information about T, do not apply]" if self.separable
                else "  [AT BOUND - do not trust]" if self.at_bound else "")
        return (
            f"T={self.temperature:.3f}  NLL {self.nll_before:.4f} -> "
            f"{self.nll_after:.4f}{flag}"
        )


@dataclass(frozen=True)
class Report:
    count: int
    accuracy: float
    ece: float
    brier: float
    reliability: list[Bin]
    selective: list[SelectivePoint]
    temperature: float

    def summary(self) -> str:
        return (
            f"n={self.count}  acc={self.accuracy:.3f}  ECE={self.ece:.4f}  "
            f"Brier={self.brier:.4f}  T={self.temperature:.3f}"
        )


def _apply(predictions: Sequence[Prediction], temperature: float,
           floor: float = DEFAULT_FLOOR) -> list[Prediction]:
    if temperature == 1.0:
        return list(predictions)
    out = []
    for p in predictions:
        keys = list(p.probabilities)
        scaled = sharpen([p.probabilities[k] for k in keys], temperature, floor)
        out.append(Prediction(dict(zip(keys, scaled)), p.label))
    return out


def report(
    predictions: Sequence[Prediction],
    bins: int = 10,
    confidence: str = "max_prob",
    temperature: float = 1.0,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    floor: float = DEFAULT_FLOOR,
) -> Report:
    """Measure calibration.

    `confidence` picks WHICH scalar is being judged. Default "max_prob" is the
    textbook reliability-diagram definition; pass "margin" to measure the number a
    confidence-gated router actually gates on, which is not the same curve.
    """
    if not predictions:
        raise ValueError("cannot report on an empty sample")

    scaled = _apply(predictions, temperature, floor)
    confidences, correct, dists, labels = [], [], [], []
    for p in scaled:
        values = list(p.probabilities.values())
        keys = list(p.probabilities)
        confidences.append(confidence_from(values, confidence))
        correct.append(keys[max(range(len(values)), key=values.__getitem__)] == p.label)
        dists.append(dict(p.probabilities))
        labels.append(p.label)

    return Report(
        count=len(scaled),
        accuracy=sum(correct) / len(correct),
        ece=expected_calibration_error(confidences, correct, bins),
        brier=brier_score(dists, labels),
        reliability=reliability_bins(confidences, correct, bins),
        selective=selective_accuracy(confidences, correct, thresholds),
        temperature=temperature,
    )


def _as_rows(predictions: Sequence[Prediction], floor: float = DEFAULT_FLOOR):
    rows, labels = [], []
    for p in predictions:
        keys = list(p.probabilities)
        if p.label not in p.probabilities:
            raise ValueError(f"label {p.label!r} is not among the options {sorted(keys)}")
        # the same floor sharpen() applies, so the fitted T is the one that gets used
        rows.append([math.log(q) for q in floored([p.probabilities[k] for k in keys], floor)])
        labels.append(keys.index(p.label))
    return rows, labels


def diagnose_temperature(
    predictions: Sequence[Prediction],
    bounds: tuple[float, float] = (0.001, 20.0),
    rtol: float = 0.05,
    floor: float = DEFAULT_FLOOR,
) -> TemperatureFit:
    """Fit a temperature and say whether the result is trustworthy.

    A temperature pinned against a search bound is a degenerate fit: the data wants a
    more extreme scaling than the search allows, which in practice means too few
    examples. Treat it as "calibration failed", not as a calibrated model.
    """
    if not predictions:
        raise ValueError("cannot fit on an empty sample")

    rows, labels = _as_rows(predictions, floor)
    t = fit_temperature(rows, labels, bounds)
    lo, hi = bounds
    separable = all(row.index(max(row)) == label for row, label in zip(rows, labels))
    return TemperatureFit(
        temperature=t,
        at_bound=(t <= lo * (1 + rtol)) or (t >= hi * (1 - rtol)),
        nll_before=negative_log_likelihood(rows, labels, 1.0),
        nll_after=negative_log_likelihood(rows, labels, t),
        separable=separable,
    )


def fit_temperature_from_predictions(
    predictions: Sequence[Prediction],
    bounds: tuple[float, float] = (0.001, 20.0),
    floor: float = DEFAULT_FLOOR,
) -> float:
    """Fit the temperature that best explains these outcomes, by NLL.

    Works in log-probability space so it composes after permutation averaging and
    contextual debiasing, both of which destroy the original logits.
    """
    if not predictions:
        raise ValueError("cannot fit on an empty sample")
    rows, labels = _as_rows(predictions, floor)
    return fit_temperature(rows, labels, bounds)
