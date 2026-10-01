"""The promotion gate: may a distilled head replace the teacher for one question family?

Both models are scored on the SAME held-out rows, so the comparison is paired: resample
rows (not models) and look at the distribution of the per-row difference. A head is
promoted only if, at the chosen confidence level, it is

    * not worse on Brier by more than `brier_margin`, and
    * not worse on accuracy by more than `max_accuracy_drop`.

Non-inferiority on both, rather than "Brier point estimate is lower": the point-estimate
gate promoted a head that was better calibrated but less accurate, on 22 rows where
neither difference meant anything. With a confidence interval, too little evidence
fails on its own; `min_eval` is only a floor so a handful of rows cannot pass by luck.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class GatePolicy:
    min_eval: int = 30
    brier_margin: float = 0.02
    max_accuracy_drop: float = 0.02
    confidence: float = 0.95
    resamples: int = 2000
    seed: int = 0


def _brier(dist: Sequence[float], outcome: int) -> float:
    return sum((p - (1.0 if i == outcome else 0.0)) ** 2 for i, p in enumerate(dist))


def _correct(dist: Sequence[float], outcome: int) -> float:
    return 1.0 if max(range(len(dist)), key=dist.__getitem__) == outcome else 0.0


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def paired_interval(diffs: Sequence[float], policy: GatePolicy) -> tuple[float, float]:
    """Two-sided percentile bootstrap interval for the mean of per-row differences."""
    rng = random.Random(policy.seed)
    n = len(diffs)
    means = sorted(_mean([diffs[rng.randrange(n)] for _ in range(n)])
                   for _ in range(policy.resamples))
    tail = (1.0 - policy.confidence) / 2.0
    lo = means[int(tail * (policy.resamples - 1))]
    hi = means[int((1.0 - tail) * (policy.resamples - 1))]
    return lo, hi


def compare(student: Sequence[Sequence[float]], teacher: Sequence[Sequence[float]],
            outcomes: Sequence[int], policy: GatePolicy = GatePolicy()) -> dict:
    """Verdict + evidence, ready to store in head.json."""
    n = len(outcomes)
    if not (len(student) == len(teacher) == n):
        raise ValueError("student, teacher and outcomes must be aligned")
    if n == 0:
        return {"promoted": False, "reasons": ["no held-out outcomes"], "n_eval": 0,
                "policy": asdict(policy)}

    s_brier = [_brier(d, o) for d, o in zip(student, outcomes)]
    t_brier = [_brier(d, o) for d, o in zip(teacher, outcomes)]
    s_acc = [_correct(d, o) for d, o in zip(student, outcomes)]
    t_acc = [_correct(d, o) for d, o in zip(teacher, outcomes)]

    brier_ci = paired_interval([s - t for s, t in zip(s_brier, t_brier)], policy)
    acc_ci = paired_interval([s - t for s, t in zip(s_acc, t_acc)], policy)

    reasons = []
    if n < policy.min_eval:
        reasons.append(f"only {n} held-out outcomes (< {policy.min_eval})")
    if brier_ci[1] > policy.brier_margin:
        reasons.append(f"Brier may be worse than teacher: diff CI upper {brier_ci[1]:+.3f} "
                       f"> {policy.brier_margin:+.3f}")
    if acc_ci[0] < -policy.max_accuracy_drop:
        reasons.append(f"accuracy may drop: diff CI lower {acc_ci[0]:+.3f} "
                       f"< {-policy.max_accuracy_drop:+.3f}")

    return {
        "promoted": not reasons,
        "reasons": reasons or ["ok"],
        "n_eval": n,
        "student_brier": _mean(s_brier), "teacher_brier": _mean(t_brier),
        "student_accuracy": _mean(s_acc), "teacher_accuracy": _mean(t_acc),
        "brier_diff_ci": list(brier_ci), "accuracy_diff_ci": list(acc_ci),
        # paired disagreements: who is right when they differ
        "student_only_right": int(sum(s > t for s, t in zip(s_acc, t_acc))),
        "teacher_only_right": int(sum(t > s for s, t in zip(s_acc, t_acc))),
        "policy": asdict(policy),
    }
