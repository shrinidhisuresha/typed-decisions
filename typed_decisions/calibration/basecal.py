"""BaseCal (design doc section 9): borrow the base model's probabilities.

Instruct tuning makes a model follow the rubric and, as a side effect, overconfident --
most overconfident exactly where it is wrong. The -Base twin at the same size has the
same tokenizer and far better calibrated next-token probabilities, but follows
instructions less reliably. So pool the two:

    log p  =  (1 - w) * log p_instruct  +  w * log p_base      (then renormalise)

w = 0 is instruct alone, w = 1 is base alone. A log-linear pool rather than an average,
because it is what "use the base model's confidence" means on the log scale the models
actually produce, and it keeps a near-zero from either side near zero.

The weight is an empirical question on your data -- see examples/basecal.py. Pool
BEFORE temperature scaling, so the fitted temperature corrects the combination.
"""

from __future__ import annotations

import math
from typing import Sequence

_FLOOR = 1e-12


def pool(instruct: Sequence[float], base: Sequence[float], weight: float) -> list[float]:
    if len(instruct) != len(base):
        raise ValueError(f"distributions differ in length: {len(instruct)} vs {len(base)}")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("basecal weight must be in [0, 1]")
    if weight == 0.0:
        return list(instruct)
    if weight == 1.0:
        return list(base)
    logs = [(1.0 - weight) * math.log(max(p, _FLOOR)) + weight * math.log(max(q, _FLOOR))
            for p, q in zip(instruct, base)]
    top = max(logs)
    exps = [math.exp(x - top) for x in logs]
    total = sum(exps)
    return [e / total for e in exps]
