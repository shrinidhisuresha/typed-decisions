"""Render prompts as prefix + branch.

Split deliberately: render_prefix() holds the state and nothing else, render_branch()
holds one question and nothing else. Callers physically cannot put the question before
the state, which is what keeps the KV cache shareable across every question in a batch
(design doc section 5).
"""

from __future__ import annotations

import json

from .calibration.permutation import Ordering
from .labels import LABEL_ALPHABET
from .types import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion

NOUL_OPTIONS = ("yes", "no")


def state_body(state: object) -> str:
    """How a state is written into the prompt (and handed to retrievers and students)."""
    return state if isinstance(state, str) else json.dumps(state, indent=2, sort_keys=True)


def render_prefix(state: object) -> str:
    """The shared, cacheable half. State only -- never a question."""
    body = state_body(state)
    return (
        "You are a decision engine. Read the state below, then answer the question "
        "that follows with a single letter.\n\n"
        "<state>\n"
        f"{body}\n"
        "</state>\n"
    )


def describe(question: Question, option: str) -> str | None:
    """The rubric text for one option, if the question gives one."""
    if isinstance(question, ChoiceQuestion):
        return question.descriptions.get(option)
    if isinstance(question, NoulQuestion) and question.criteria:
        return question.criteria.get("true" if option == "yes" else "false")
    return None


def option_line(question: Question, option: str) -> str:
    """`option` or `option: description` -- what the model (and the retriever) read."""
    text = describe(question, option)
    return f"{option}: {text}" if text else option


def _render_options(instructions: str, options: list[str]) -> str:
    lines = [f"{letter}) {opt}" for letter, opt in zip(LABEL_ALPHABET, options)]
    listing = "\n".join(lines)
    return f"\n<question>\n{instructions}\n{listing}\n</question>\nAnswer:"


def declared_options(question: Question) -> list[str]:
    """The option texts in the order the caller declared them."""
    if isinstance(question, ChoiceQuestion):
        return list(question.criteria)
    if isinstance(question, NoulQuestion):
        return list(NOUL_OPTIONS)
    if isinstance(question, ScoreQuestion):
        return list(question.criteria)
    raise TypeError(f"unsupported question type: {type(question).__name__}")


def options_in_order(question: Question, ordering: Ordering | None) -> list[str]:
    """Option texts arranged for display. ordering[i] is the ORIGINAL index at position i."""
    options = declared_options(question)
    if ordering is None:
        return options
    if sorted(ordering) != list(range(len(options))):
        raise ValueError(
            f"ordering {ordering} is not a permutation of {len(options)} options"
        )
    return [options[original] for original in ordering]


def render_sequence_branch(question: Question, ordering: Ordering | None = None) -> str:
    """For sequence scoring: options listed without letters; the answer is the option text
    itself, scored as a continuation after "Answer:"."""
    listing = "\n".join(f"- {option_line(question, opt)}"
                        for opt in options_in_order(question, ordering))
    return (f"\n<question>\n{question.instructions}\nReply with one option, exactly as "
            f"written.\n{listing}\n</question>\nAnswer:")


def render_branch(question: Question, ordering: Ordering | None = None) -> str:
    """The per-question half. One question, in isolation, never naming another."""
    return _render_options(question.instructions,
                           [option_line(question, o) for o in options_in_order(question, ordering)])

