"""Turn raw model logits into typed, calibrated answers (design doc section 2)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


CONFIDENCE_MODES = ("margin", "max_prob", "entropy", "normalized_max")


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float


def _softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    m = max(logits)
    exps = [math.exp((z - m) / temperature) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def confidence_from(probs: Sequence[float], mode: str = "margin") -> float:
    """Collapse a distribution into a single confidence scalar.

    margin    top1 - top2. Best default for a confidence-gated router.
    max_prob  the winning probability. Most interpretable.
    entropy   1 - H(p)/log N. Comparable across different option counts.
    normalized_max  (N*max - 1)/(N - 1): 0 at uniform, 1 when certain; max_prob rescaled
              so it is comparable across option counts.
    """
    if mode not in CONFIDENCE_MODES:
        raise ValueError(f"unknown confidence mode {mode!r}; expected one of {CONFIDENCE_MODES}")
    if mode == "max_prob":
        return max(probs)
    if mode == "margin":
        top = sorted(probs, reverse=True)
        return top[0] - (top[1] if len(top) > 1 else 0.0)
    n = len(probs)
    if mode == "normalized_max":
        return 1.0 if n < 2 else max(0.0, (n * max(probs) - 1.0) / (n - 1))
    if n < 2:
        return 1.0
    h = -sum(p * math.log(p) for p in probs if p > 0.0)
    return 1.0 - h / math.log(n)


def choice_from_probs(
    probabilities: Sequence[float],
    options: Sequence[str],
    confidence: str = "margin",
) -> ChoiceAnswer:
    """Build a Choice answer from an already-final distribution.

    This is the primitive the calibrated pipeline uses: permutation averaging and
    contextual debiasing both work in probability space, so by the time we get here
    the logits are long gone.
    """
    if len(probabilities) != len(options):
        raise ValueError("probabilities and options must be the same length")
    by_option = dict(zip(options, probabilities))
    return ChoiceAnswer(
        choice=max(by_option, key=by_option.__getitem__),
        probabilities=by_option,
        confidence=confidence_from(probabilities, confidence),
    )


def score_choice(
    option_logits: Sequence[float],
    options: Sequence[str],
    temperature: float = 1.0,
    confidence: str = "margin",
) -> ChoiceAnswer:
    """Uncalibrated convenience path: softmax the raw logits, then build the answer."""
    return choice_from_probs(_softmax(option_logits, temperature), options, confidence)


@dataclass(frozen=True)
class NoulAnswer:
    noul: float


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    legend: dict[str, float]
    probabilities: dict[str, float]
    confidence: float


def noul_from_probs(p_yes: float) -> NoulAnswer:
    """A named truth value in [0,1]. The number IS the probability, deliberately
    not thresholded -- each call site picks its own cutoff."""
    return NoulAnswer(noul=p_yes)


def score_noul(yes_logit: float, no_logit: float, temperature: float = 1.0) -> NoulAnswer:
    p_yes, _p_no = _softmax([yes_logit, no_logit], temperature)
    return noul_from_probs(p_yes)


def score_score(
    band_logits: Sequence[float],
    legend: dict[str, float],
    temperature: float = 1.0,
    confidence: str = "margin",
) -> ScoreAnswer:
    """Ordinal bands collapsed to a continuous score by expectation (the G-Eval trick).

    The model never types a number; it distributes mass over labelled bands and we
    take sum(p_i * value_i). Far more stable than asking for "7.4" directly.
    """
    return score_from_probs(_softmax(band_logits, temperature), legend, confidence)


def score_from_probs(
    probabilities: Sequence[float],
    legend: dict[str, float],
    confidence: str = "margin",
) -> ScoreAnswer:
    """Collapse an already-final band distribution to a continuous score."""
    if len(legend) < 2:
        raise ValueError("legend needs at least two bands to score against")
    labels = list(legend)
    by_label = dict(zip(labels, probabilities))
    return ScoreAnswer(
        score=sum(by_label[label] * legend[label] for label in labels),
        legend=dict(legend),
        probabilities=by_label,
        confidence=confidence_from(probabilities, confidence),
    )
