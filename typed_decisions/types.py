"""Typed questions: {type, instructions, criteria}."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChoiceQuestion:
    instructions: str
    criteria: list[str]
    type: str = field(default="choice", init=False)
    # Optional rubric text per option (`{option: description}` on the wire); rendered next to the
    # option in the prompt and used as the retrieval document. Absent means the option name
    # stands alone.
    descriptions: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class NoulQuestion:
    instructions: str
    type: str = field(default="noul", init=False)
    # Optional meaning of a yes and a no (`{"true": ..., "false": ...}` on the wire).
    criteria: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoreQuestion:
    instructions: str
    criteria: dict[str, float]
    type: str = field(default="score", init=False)


Question = ChoiceQuestion | NoulQuestion | ScoreQuestion


def question_from_dict(data: dict) -> Question:
    """Parse one question from the wire shape: {type, instructions, criteria}."""
    if not isinstance(data, dict):
        raise ValueError("a question must be an object with a 'type'")
    kind = data.get("type")
    instructions = data.get("instructions")
    if not isinstance(instructions, str) or not instructions:
        raise ValueError("question needs non-empty 'instructions'")
    criteria = data.get("criteria")
    if kind == "choice":
        if isinstance(criteria, dict) and len(criteria) >= 2:
            options = [str(k) for k in criteria]
            described = {str(k): str(v) for k, v in criteria.items() if v not in (None, "")}
            return ChoiceQuestion(instructions, options, descriptions=described)
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("choice needs 'criteria': a list of at least two options, or an "
                             "{option: description} object")
        return ChoiceQuestion(instructions, [str(c) for c in criteria])
    if kind == "noul":
        if criteria is None:
            return NoulQuestion(instructions)
        if not isinstance(criteria, dict) or set(criteria) - {"true", "false"}:
            raise ValueError("noul 'criteria' must be an object with 'true' and/or 'false'")
        return NoulQuestion(instructions, criteria={k: str(v) for k, v in criteria.items()})
    if kind == "score":
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError("score needs 'criteria': a {label: value} legend of 2+ entries")
        return ScoreQuestion(instructions, {str(k): float(v) for k, v in criteria.items()})
    raise ValueError(f"unknown question type {kind!r}; expected choice, noul or score")
