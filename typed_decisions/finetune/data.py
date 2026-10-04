"""Training rows: many question families from one labelled dataset, in random option order.

A single labelled dataset becomes many schemas -- random option subsets, reworded
instructions, Noul rephrasings -- so the adapter learns to score *a* typed question
calibratedly rather than memorising one label set. Evaluation must use datasets that never
appear here (eval.py), because zero-shot calibration on new schemas is the whole claim.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from ..calibration.contextual import NULL_STATE
from ..labels import allocate_labels
from ..prompt import declared_options, render_branch, render_prefix
from ..types import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion

CHOICE_INSTRUCTIONS = (
    "What does the customer want?",
    "Which of these best describes the request?",
    "Classify this message.",
    "What is this message about?",
    "Pick the intent that matches the message.",
    "Which topic does this belong to?",
)

NOUL_INSTRUCTIONS = (
    "Is the customer asking about {x}?",
    "Is this message about {x}?",
    "Does this request concern {x}?",
)


@dataclass(frozen=True)
class Row:
    state: object
    question: Question
    target: tuple[float, ...]      # over declared_options(question), in display order


def _choice(state, label: str, labels: Sequence[str], rng: random.Random,
            k_range: tuple[int, int], instructions: Sequence[str] = CHOICE_INSTRUCTIONS) -> Row:
    k = min(rng.randint(*k_range), len(labels))
    options = [label] + rng.sample([l for l in labels if l != label], k - 1)
    rng.shuffle(options)                      # the label lands in a uniform random position
    question = ChoiceQuestion(rng.choice(instructions), options)
    return Row(state, question, tuple(1.0 if o == label else 0.0 for o in options))


def _noul(state, label: str, labels: Sequence[str], rng: random.Random,
          templates: Sequence[str] = NOUL_INSTRUCTIONS) -> Row:
    asked = label if rng.random() < 0.5 else rng.choice([l for l in labels if l != label])
    question = NoulQuestion(rng.choice(templates).format(x=asked))
    return Row(state, question, (1.0, 0.0) if asked == label else (0.0, 1.0))


def _native_noul(state, label: str, labels: Sequence[str], rng: random.Random,
                 questions: Sequence[str], negated: Sequence[str]) -> Row:
    """A dataset's own binary question. Negated phrasings flip which label is "yes", so a
    standing preference for answering yes is penalised rather than rewarded."""
    if negated and rng.random() < 0.35:
        question, yes = NoulQuestion(rng.choice(negated)), labels[1]
    else:
        question, yes = NoulQuestion(rng.choice(questions)), labels[0]
    return Row(state, question, (1.0, 0.0) if label == yes else (0.0, 1.0))


def _score(state, label: str, levels: Sequence[str], rng: random.Random,
           questions: Sequence[str]) -> Row:
    """An ordinal legend, value i+1 for level i. Shown in natural order half the time and
    shuffled otherwise, so position carries no information about the level."""
    shown = list(levels)
    if rng.random() < 0.5:
        rng.shuffle(shown)
    question = ScoreQuestion(rng.choice(questions),
                             {lvl: float(levels.index(lvl) + 1) for lvl in shown})
    return Row(state, question, tuple(1.0 if lvl == label else 0.0 for lvl in shown))


def _null(row: Row) -> Row:
    """The same question over a content-free state. Every family here is balanced, so
    the prior is uniform: with no evidence the right answer is the base rate, not
    whichever letter the model happens to like."""
    n = len(row.target)
    return Row(NULL_STATE, row.question, tuple([1.0 / n] * n))


def build(samples: Sequence, labels: Sequence[str], seed: int = 0, noul_share: float = 0.3,
          null_share: float = 0.1, k_range: tuple[int, int] = (3, 12)) -> list[Row]:
    """One row per sample (Choice or Noul), plus null-state rows."""
    rng = random.Random(seed)
    rows = []
    for s in samples:
        row = (_noul(s.state, s.label, labels, rng) if rng.random() < noul_share
               else _choice(s.state, s.label, labels, rng, k_range))
        rows.append(row)
        if rng.random() < null_share:
            rows.append(_null(row))
    rng.shuffle(rows)
    return rows


def build_mixture(names: Sequence[str], seed: int = 0, null_share: float = 0.1,
                  about_share: float = 0.3, k_range: tuple[int, int] = (2, 12),
                  cache_dir: str = "data", scale: float = 1.0) -> list[Row]:
    """Rows from every source in `names`, each in its own question shape(s)."""
    from .sources import SOURCES, balanced, rows_for

    rng = random.Random(seed)
    rows: list[Row] = []
    for name in names:
        src = SOURCES[name]
        raw = rows_for(src, cache_dir)
        per_class = max(1, round(src.per_class * scale))
        if src.shape == "choice":
            for s in balanced(src, raw, src.labels, per_class, seed=seed):
                if rng.random() < about_share:
                    rows.append(_noul(s.state, s.label, src.labels, rng,
                                      src.about_questions or NOUL_INSTRUCTIONS))
                else:
                    rows.append(_choice(s.state, s.label, src.labels, rng, k_range,
                                        src.choice_questions or CHOICE_INSTRUCTIONS))
        elif src.shape == "noul":
            for s in balanced(src, raw, src.labels, per_class, seed=seed):
                rows.append(_native_noul(s.state, s.label, src.labels, rng,
                                         src.noul_questions, src.negated_questions))
        elif src.shape == "score":
            # Two legends from one dataset: the full scale, and the collapsed one, each
            # sampled balanced on ITS OWN levels so the null-state prior stays uniform.
            for s in balanced(src, raw, src.labels, per_class, seed=seed):
                rows.append(_score(s.state, s.label, src.labels, rng, src.score_questions))
            if src.collapse:
                coarse = list(dict.fromkeys(src.collapse.values()))
                for s in balanced(src, raw, coarse, per_class, relabel=src.collapse.get,
                                  seed=seed):
                    rows.append(_score(s.state, s.label, coarse, rng, src.score_questions))
        elif src.shape == "soft_noul":
            # Stratified over the label buckets (deciles of P(yes)), so ambiguous rows are
            # not drowned by the clear-cut majority.
            for s in balanced(src, raw, src.labels, per_class, seed=seed):
                rows.append(Row(s.state, NoulQuestion(rng.choice(src.noul_questions)),
                                (s.soft, 1.0 - s.soft)))
        else:
            raise ValueError(f"{name}: unknown shape {src.shape!r}")
    rows += [_null(r) for r in rows if rng.random() < null_share]
    rng.shuffle(rows)
    return rows


@dataclass(frozen=True)
class Encoded:
    input_ids: list[int]
    label_ids: list[int]
    target: list[float]
    # Ordinal levels in their natural order, as positions in label_ids; set for Score rows
    # so the loss can add RPS. Empty otherwise.
    ordinal: tuple[int, ...] = ()


def encode(row: Row, tokenizer) -> Encoded:
    """The exact prompt the client renders for one question, ordering=None (the row's
    options are already shuffled), plus the label tokens the client would read."""
    text = render_prefix(row.state) + render_branch(row.question)
    options = declared_options(row.question)
    labels = allocate_labels(options, tokenizer)
    ordinal: tuple[int, ...] = ()
    if isinstance(row.question, ScoreQuestion):
        values = row.question.criteria
        ordinal = tuple(sorted(range(len(options)), key=lambda i: values[options[i]]))
    return Encoded(tokenizer.encode(text), [l.token_id for l in labels], list(row.target),
                   ordinal)
