"""Map answer options onto distinct single tokens.

The whole logit-scoring trick (design doc section 2) depends on every option being
readable at ONE position in the logit vector. That is a tokenizer property, not an
assumption -- so we assert it here and fail loudly rather than silently scoring
the first token of a multi-token label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

LABEL_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class LabelError(ValueError):
    """Options could not be mapped onto distinct single tokens."""


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class Label:
    option: str
    label: str
    rendered: str
    token_id: int


def allocate_labels(options: Sequence[str], tokenizer: Tokenizer) -> list[Label]:
    if len(options) < 1:
        raise LabelError("a choice needs at least one option")
    if len(set(options)) != len(options):
        raise LabelError("duplicate options are ambiguous; each option must be distinct")
    if len(options) > len(LABEL_ALPHABET):
        raise LabelError(
            f"{len(options)} options exceeds the {len(LABEL_ALPHABET)}-label alphabet; "
            "use the retrieval stage (design doc section 3C) for high-cardinality choices"
        )

    labels: list[Label] = []
    for option, letter in zip(options, LABEL_ALPHABET):
        rendered = f" {letter}"
        ids = tokenizer.encode(rendered)
        if len(ids) != 1:
            raise LabelError(
                f"label {rendered!r} is not a single token for this tokenizer "
                f"(got {len(ids)} tokens); logit scoring needs one token per option"
            )
        labels.append(Label(option=option, label=letter, rendered=rendered, token_id=ids[0]))

    ids = [l.token_id for l in labels]
    if len(set(ids)) != len(ids):
        raise LabelError("tokenizer mapped two labels onto the same token id")
    return labels
