"""Traffic + outcome log: the raw material for Phase 3 distillation (design doc section 7).

Every ask() the server answers is appended as one JSON line; outcomes arrive later,
often much later, keyed by the request id the ask returned. `examples()` joins the two
into per-family training rows.

A *family* is one fixed question definition -- same type, instructions and options --
because a distilled head has a fixed output layer. Rewording a question starts a new
family; that is deliberate, since the old labels no longer mean the same thing.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .prompt import declared_options
from .types import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion, question_from_dict


TEACHER_SOURCE = "teacher"


def question_to_dict(question: Question) -> dict:
    data = {"type": question.type, "instructions": question.instructions}
    if isinstance(question, ChoiceQuestion) and question.descriptions:
        data["criteria"] = {o: question.descriptions.get(o) for o in question.criteria}
    elif isinstance(question, (ChoiceQuestion, ScoreQuestion)):
        data["criteria"] = question.criteria
    elif isinstance(question, NoulQuestion) and question.criteria:
        data["criteria"] = dict(question.criteria)
    return data


def family_key(question: Question) -> str:
    # Key order is meaningful: it is the option order, and so the head's output order.
    # Never sort_keys here or in the log, or a score legend silently reorders.
    canonical = json.dumps(question_to_dict(question))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def teacher_distribution(question: Question, answer: dict) -> list[float]:
    """The teacher's distribution over declared_options(question), from a wire answer."""
    if isinstance(question, NoulQuestion):
        p = float(answer["noul"])
        return [p, 1.0 - p]
    probs = answer["probabilities"]
    return [float(probs[o]) for o in declared_options(question)]


class TrafficLog:
    """Append-only JSONL. Thread-safe within one process, which is all the server needs."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _append(self, record: dict) -> None:
        line = json.dumps(record)
        with self._lock, self.path.open("a") as fh:
            fh.write(line + "\n")

    def record_ask(self, state, questions: dict[str, Question], response: dict,
                   model: str, request_id: str | None = None) -> str:
        """`response` may carry `served_by` (from routing.Router); absent means teacher."""
        request_id = request_id or uuid.uuid4().hex
        self._append({
            "served_by": response.get("served_by") or {},
            "shadow": response.get("shadow") or {},
            "kind": "ask",
            "id": request_id,
            "ts": time.time(),
            "model": model,
            "state": state,
            "questions": {name: question_to_dict(q) for name, q in questions.items()},
            "answers": response["answers"],
        })
        return request_id

    def record_outcome(self, request_id: str, outcomes: dict) -> None:
        if not isinstance(outcomes, dict) or not outcomes:
            raise ValueError("'outcomes' must be a non-empty {question name: label} object")
        self._append({"kind": "outcome", "id": request_id, "ts": time.time(),
                      "outcomes": outcomes})

    def records(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with self.path.open() as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


@dataclass(frozen=True)
class Example:
    """One (state, question) pair: what the teacher said, and what actually happened."""

    request_id: str
    family: str
    question: Question
    state: object
    teacher: list[float]          # the answering model's distribution, over declared_options
    outcome: int | None           # index into declared_options, if known
    source: str = TEACHER_SOURCE  # "teacher", or "head:<family>" if a head answered
    shadow: dict | None = None    # {head, version, probabilities} if a head ran in shadow

    @property
    def options(self) -> list[str]:
        return declared_options(self.question)


def _outcome_index(question: Question, value) -> int:
    options = declared_options(question)
    if isinstance(question, NoulQuestion):
        if isinstance(value, bool):
            return 0 if value else 1
        if value in ("yes", "no"):
            return options.index(value)
        raise ValueError(f"noul outcome must be true/false or yes/no, got {value!r}")
    if value not in options:
        raise ValueError(f"outcome {value!r} is not one of {options}")
    return options.index(value)


def examples(log: TrafficLog, family: str | None = None) -> list[Example]:
    """Join asks with their outcomes. The latest outcome for a (request, question) wins.

    An outcome naming an unknown request or question is skipped -- it may belong to a
    log that was rotated -- but one with an invalid label raises, because that is a bug
    in whoever is reporting outcomes.
    """
    asks: dict[str, dict] = {}
    outcomes: dict[tuple[str, str], object] = {}
    for record in log.records():
        if record["kind"] == "ask":
            asks[record["id"]] = record
        elif record["kind"] == "outcome":
            for name, value in record["outcomes"].items():
                outcomes[(record["id"], name)] = value

    out = []
    for request_id, ask in asks.items():
        for name, qdict in ask["questions"].items():
            question = question_from_dict(qdict)
            key = family_key(question)
            if family is not None and key != family:
                continue
            raw = outcomes.get((request_id, name))
            out.append(Example(
                request_id=request_id,
                family=key,
                question=question,
                state=ask["state"],
                teacher=teacher_distribution(question, ask["answers"][name]),
                outcome=None if raw is None else _outcome_index(question, raw),
                source=ask.get("served_by", {}).get(name, TEACHER_SOURCE),
                shadow=ask.get("shadow", {}).get(name),
            ))
    return out
