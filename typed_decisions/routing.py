"""Serve distilled heads where they have earned it, and the teacher everywhere else.

For each question in an ask: if a promoted Head exists for its family, the head answers
(no decoder pass at all); otherwise -- or if the head is below `min_confidence` -- the
question goes to the teacher. All teacher-bound questions still share one prefix pass.

`served_by` in the response says which model answered each question. The traffic log
records it too, because a head's answer must never become a head's training target:
that loop would distil the student into itself and freeze its mistakes.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Mapping, Sequence

from .client import Decider, Usage
from .distill import Head
from .scoring import confidence_from
from .traffic import TEACHER_SOURCE as TEACHER, family_key
from .types import Question


class Router:
    def __init__(self, client: Decider, heads: Sequence[Head] = (), min_confidence: float = 0.0,
                 shadow_heads: Sequence[Head] = ()):
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        self.client = client
        self.heads: dict[str, Head] = {}
        for head in heads:
            if head.family in self.heads:
                raise ValueError(f"two heads for family {head.family}")
            self.heads[head.family] = head
        self.min_confidence = min_confidence
        # Shadow heads run on teacher-served questions; their answers are logged for
        # evaluation on real outcomes and never returned. A served head wins over a
        # shadow head for the same family.
        self.shadow_heads = {h.family: h for h in shadow_heads if h.family not in self.heads}

    def ask(self, state: object, questions: Mapping[str, Question]) -> dict:
        """Same response shape as Result.to_dict(), plus `served_by`."""
        if not questions:
            raise ValueError("ask() needs at least one question")
        answers: dict[str, dict] = {}
        served_by: dict[str, str] = {}
        to_teacher: dict[str, Question] = {}

        for name, question in questions.items():
            head = self.heads.get(family_key(question))
            if head is not None:
                probs = head.predict_proba([state])[0]
                if confidence_from(probs, "margin") >= self.min_confidence:
                    answers[name] = asdict(head.to_answer(probs))
                    served_by[name] = head.label
                    continue
            to_teacher[name] = question

        shadow: dict[str, dict] = {}
        for name, question in to_teacher.items():
            head = self.shadow_heads.get(family_key(question))
            if head is not None:
                shadow[name] = {"head": head.label, "version": head.version,
                                "probabilities": head.predict_proba([state])[0]}

        usage = Usage(input_tokens=0, output_tokens=0)
        if to_teacher:
            result = self.client.ask(state, to_teacher)
            for name, answer in result.answers.items():
                answers[name] = asdict(answer)
                served_by[name] = TEACHER
            usage = result.usage

        response = {
            "answers": {name: answers[name] for name in questions},  # caller's order
            "usage": asdict(usage),
            "served_by": {name: served_by[name] for name in questions},
        }
        if shadow:
            response["shadow"] = shadow   # for the log; the server strips it
        return response


def load_heads(directory, device: str | None = None, include_unpromoted: bool = False):
    """Every head under `directory`; (served, skipped-with-reason)."""
    from pathlib import Path
    served, skipped = [], []
    for meta in sorted(Path(directory).glob("*/head.json")):
        head = Head.load(meta.parent, device=device)
        if head.promoted or include_unpromoted:
            served.append(head)
        else:
            reasons = head.evaluation.get("reasons") or [head.evaluation.get("reason", "not evaluated")]
            skipped.append((head.family, "; ".join(reasons)))
    return served, skipped
