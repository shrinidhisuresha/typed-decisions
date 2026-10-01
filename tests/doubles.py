"""Test doubles. Deliberately not shipped in typed_decisions/."""

from __future__ import annotations

from typed_decisions.labels import LABEL_ALPHABET


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        if text.startswith(" ") and len(text) == 2 and text[1] in LABEL_ALPHABET:
            return [1000 + LABEL_ALPHABET.index(text[1])]
        return [7] * max(len(text), 1)


class RecordingBackend:
    """Returns canned logits, and records exactly what it was asked to score."""

    def __init__(self, logits_by_marker: dict[str, list[float]]):
        self.logits_by_marker = logits_by_marker
        self.tokenizer = FakeTokenizer()
        self.calls: list[tuple[str, list]] = []

    def score(self, prefix, branches):
        self.calls.append((prefix, list(branches)))
        out = []
        for branch in branches:
            for marker, logits in self.logits_by_marker.items():
                if marker in branch.text:
                    out.append(logits[: branch.n_options])
                    break
            else:
                raise AssertionError(f"no canned logits for branch: {branch.text!r}")
        return out


class PositionBiasedBackend:
    """Ignores content entirely and always favours whatever sits in position 0.

    Exactly the pathology option-order permutation exists to cancel.
    """

    def __init__(self, bias: float = 2.0):
        self.bias = bias
        self.tokenizer = FakeTokenizer()
        self.calls: list[tuple[str, list]] = []

    def score(self, prefix, branches):
        self.calls.append((prefix, list(branches)))
        return [
            [self.bias if i == 0 else 0.0 for i in range(b.n_options)]
            for b in branches
        ]


class PriorOnlyBackend:
    """Returns the same distribution no matter what the state says.

    Its answers carry zero information, so contextual calibration must reduce them
    to uniform.
    """

    def __init__(self, logits_by_position: list[float]):
        self.logits = logits_by_position
        self.tokenizer = FakeTokenizer()
        self.calls: list[tuple[str, list]] = []

    def score(self, prefix, branches):
        self.calls.append((prefix, list(branches)))
        return [list(self.logits[: b.n_options]) for b in branches]


class FakeHead:
    """A distilled head that returns fixed probabilities; no model behind it."""

    def __init__(self, question, probs, promoted=True):
        from typed_decisions.traffic import family_key
        self.question = question
        self.family = family_key(question)
        self.probs = list(probs)
        self.base = "fake"
        self.version = "v1"
        self.label = f"head:{self.family}"
        self.evaluation = {"promoted": promoted, "reason": "ok" if promoted else "worse"}
        self.calls = 0

    @property
    def promoted(self):
        return self.evaluation["promoted"]

    def predict_proba(self, states, **_):
        self.calls += len(states)
        return [list(self.probs) for _ in states]

    def to_answer(self, probs, confidence="margin"):
        from typed_decisions.distill import Head
        return Head.to_answer(self, probs, confidence)
