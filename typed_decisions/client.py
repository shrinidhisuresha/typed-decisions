"""state + typed questions -> typed answers, in one shared-prefix pass."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping

from .backends import Backend, Branch
from .calibration.basecal import pool
from .calibration.contextual import NULL_STATE, debias
from .calibration.permutation import Ordering, average_distributions, rotations, unpermute
from .calibration.temperature import sharpen
from .labels import LABEL_ALPHABET, allocate_labels
from .prompt import (
    declared_options,
    options_in_order,
    render_branch,
    render_prefix,
    render_sequence_branch,
    option_line,
    state_body,
)
from .retrieval import Retriever, shortlist
from .scoring import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    _softmax,
    choice_from_probs,
    noul_from_probs,
    score_from_probs,
)
from .types import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion

Answer = ChoiceAnswer | NoulAnswer | ScoreAnswer

# Above the letter alphabet, a Choice is scored by option text (design doc section 2).
# A practical ceiling for one prompt's option listing; past it, use retrieval (3C).
MAX_CHOICE_OPTIONS = 255


@dataclass(frozen=True)
class Calibration:
    """Phase 1 pipeline settings (design doc section 4).

    Applied in this order, which is deliberate: the two unsupervised debiasing steps
    run first and need no labels, then the fitted temperature is applied last to
    whatever they produced.

        permutations -> contextual -> temperature

    `temperature` must come from `fit_temperature()` on held-out labelled data.
    Leaving it at 1.0 means no supervised calibration has been done, and `confidence`
    should not be gated on.
    """

    permutations: int = 1
    contextual: bool = False
    temperature: float = 1.0
    confidence: str = "margin"
    basecal: float = 0.0

    def __post_init__(self):
        if self.permutations < 1:
            raise ValueError("permutations must be at least 1")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= self.basecal <= 1.0:
            raise ValueError("basecal must be in [0, 1]")


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Result:
    answers: dict[str, Answer]
    usage: Usage

    def to_dict(self) -> dict:
        """Serialise to the wire response shape ({answers, usage}): one stable shape
        whatever the backend."""
        return {
            "answers": {name: asdict(answer) for name, answer in self.answers.items()},
            "usage": asdict(self.usage),
        }


@dataclass(frozen=True)
class _Plan:
    """One question's branches, one per ordering, and where they sit in the batch."""

    name: str
    question: Question
    orderings: list[Ordering]
    slice_: slice


def _expand(answer: ChoiceAnswer, full: ChoiceQuestion) -> ChoiceAnswer:
    """Back to the caller's full option list: options cut by retrieval get probability 0."""
    probs = {option: answer.probabilities.get(option, 0.0) for option in full.criteria}
    return ChoiceAnswer(choice=answer.choice, probabilities=probs, confidence=answer.confidence)


def _check_same_labels(backend: Backend, reference: Backend) -> None:
    """Branch token ids come from the main tokenizer; the reference must agree on
    every label token or it would be scoring different options."""
    for letter in LABEL_ALPHABET:
        text = f" {letter}"
        if backend.tokenizer.encode(text) != reference.tokenizer.encode(text):
            raise ValueError(
                f"reference tokenizer encodes {text!r} differently; BaseCal needs the "
                "-Base twin of the SAME model family"
            )


class Decider:
    def __init__(
        self,
        backend: Backend,
        calibration: Calibration | None = None,
        temperature: float = 1.0,
        reference: Backend | None = None,
        sequence_threshold: int = len(LABEL_ALPHABET),
        sequence_norm: str = "mean",
        retriever: Retriever | None = None,
        shortlist_k: int = 10,
        retrieve_above: int = len(LABEL_ALPHABET),
    ):
        # Retrieval (design doc 3C): a Choice with more than `retrieve_above` options is
        # narrowed to its `shortlist_k` best by the retriever, then scored as usual.
        if shortlist_k < 2:
            raise ValueError("shortlist_k must be at least 2")
        self.retriever = retriever
        self.shortlist_k = shortlist_k
        self.retrieve_above = retrieve_above
        if sequence_norm not in ("mean", "sum"):
            raise ValueError("sequence_norm must be 'mean' or 'sum'")
        self.sequence_norm = sequence_norm
        self.backend = backend
        # Choice questions with MORE options than this are scored by option text. The
        # default is the letter alphabet; 0 forces sequence scoring (for measuring it).
        self.sequence_threshold = sequence_threshold
        self.calibration = calibration or Calibration()
        # BaseCal: the -Base twin, scored on the same branches and pooled in.
        self.reference = reference
        if self.calibration.basecal > 0.0:
            if reference is None:
                raise ValueError("Calibration(basecal>0) needs a reference= (base) backend")
            _check_same_labels(backend, reference)
        self.raw_temperature = temperature
        # The null-state prior depends on the QUESTION, not the state, so it is
        # computed once per distinct question set and reused. Halves the cost of
        # contextual calibration in a loop.
        self._null_cache: dict[tuple[str, ...], list[list[float]]] = {}

    def ask(self, state: object, questions: Mapping[str, Question]) -> Result:
        if not questions:
            raise ValueError("ask() needs at least one question")

        originals = dict(questions)
        questions = self._narrow(state, questions)
        plans, branches = self._plan(questions)
        prefix = render_prefix(state)
        per_plan = self._debiased(self.backend, prefix, plans, branches)
        weight = self.calibration.basecal
        if weight > 0.0:
            base = self._debiased(self.reference, prefix, plans, branches)
            per_plan = [pool(p, q, weight) for p, q in zip(per_plan, base)]

        answers: dict[str, Answer] = {}
        for plan, probs in zip(plans, per_plan):
            probs = sharpen(probs, self.calibration.temperature)
            answer = self._to_answer(plan.question, probs)
            if plan.question is not originals[plan.name]:
                answer = _expand(answer, originals[plan.name])
            answers[plan.name] = answer

        encode = self.backend.tokenizer.encode
        input_tokens = len(encode(prefix)) + sum(len(encode(b.text)) for b in branches)
        return Result(answers=answers, usage=Usage(input_tokens, output_tokens=0))

    def _narrow(self, state: object, questions: Mapping[str, Question]) -> dict[str, Question]:
        """Replace each over-long Choice with its retrieved shortlist; others unchanged."""
        if self.retriever is None:
            return dict(questions)
        out: dict[str, Question] = {}
        for name, question in questions.items():
            if (isinstance(question, ChoiceQuestion)
                    and len(question.criteria) > self.retrieve_above
                    and len(question.criteria) > self.shortlist_k):
                scores = self.retriever.rank(
                    state_body(state), [option_line(question, o) for o in question.criteria],
                    task=f"Find the answer option that best answers: {question.instructions}")
                keep = shortlist(scores, self.shortlist_k)
                kept = [question.criteria[i] for i in keep]
                question = ChoiceQuestion(
                    question.instructions, kept,
                    descriptions={o: d for o, d in question.descriptions.items() if o in kept})
            out[name] = question
        return out

    def _debiased(self, backend: Backend, prefix: str, plans: list[_Plan],
                  branches: list[Branch]) -> list[list[float]]:
        """Permutation-averaged, optionally null-prior-divided distribution per plan."""
        logit_rows = backend.score(prefix, branches)
        null_rows = self._null_prior(backend, branches) if self.calibration.contextual else None
        out = []
        for plan in plans:
            probs = self._combine(plan, logit_rows[plan.slice_])
            if null_rows is not None:
                probs = debias(probs, self._combine(plan, null_rows[plan.slice_]))
            out.append(probs)
        return out

    def _null_prior(self, backend: Backend, branches: list[Branch]) -> list[list[float]]:
        key = (id(backend),) + tuple(b.text for b in branches)
        if key not in self._null_cache:
            self._null_cache[key] = backend.score(render_prefix(NULL_STATE), branches)
        return self._null_cache[key]

    def _plan(self, questions: Mapping[str, Question]) -> tuple[list[_Plan], list[Branch]]:
        plans: list[_Plan] = []
        branches: list[Branch] = []
        for name, question in questions.items():
            n = len(declared_options(question))
            if n > MAX_CHOICE_OPTIONS:
                raise ValueError(f"{name}: {n} options exceeds {MAX_CHOICE_OPTIONS}; use a "
                                 "retrieval stage (design doc section 3C)")
            by_sequence = isinstance(question, ChoiceQuestion) and n > self.sequence_threshold
            orderings = rotations(n, self.calibration.permutations)
            start = len(branches)
            for ordering in orderings:
                if by_sequence:
                    branches.append(Branch(
                        text=render_sequence_branch(question, ordering),
                        token_ids=[], rendered=[],
                        continuations=tuple(
                            f" {o}" for o in options_in_order(question, ordering)),
                        normalize=self.sequence_norm,
                    ))
                    continue
                labels = allocate_labels(
                    options_in_order(question, ordering), self.backend.tokenizer
                )
                branches.append(
                    Branch(
                        text=render_branch(question, ordering),
                        token_ids=[l.token_id for l in labels],
                        rendered=[l.rendered for l in labels],
                    )
                )
            plans.append(
                _Plan(name, question, orderings, slice(start, len(branches)))
            )
        return plans, branches

    def _combine(self, plan: _Plan, rows: list[list[float]]) -> list[float]:
        """Softmax each ordering, send every position back to its original option,
        then average. With k == n orderings this cancels positional bias exactly."""
        per_ordering = [
            unpermute(_softmax(row, self.raw_temperature), ordering)
            for row, ordering in zip(rows, plan.orderings)
        ]
        return average_distributions(per_ordering)

    def _to_answer(self, question: Question, probs: list[float]) -> Answer:
        confidence = self.calibration.confidence
        if isinstance(question, ChoiceQuestion):
            return choice_from_probs(probs, question.criteria, confidence)
        if isinstance(question, NoulQuestion):
            return noul_from_probs(probs[0])
        if isinstance(question, ScoreQuestion):
            return score_from_probs(probs, question.criteria, confidence)
        raise TypeError(f"unsupported question type: {type(question).__name__}")
