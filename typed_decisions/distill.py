"""Phase 3: distil one question family into a small encoder head (design doc sections 3B, 7).

The teacher (the decoder scorer, debiased and calibrated) answers every logged ask; the
student is an encoder with an N-way head for one fixed question family, so at serving
time the question is baked into the weights and only the state is read.

Targets, per example:
    outcome known  -> mix * one_hot(outcome) + (1 - mix) * teacher
    otherwise      -> teacher distribution
Loss is soft cross-entropy -- the log score, a proper scoring rule, so the student is
pushed toward calibrated probabilities rather than just the right argmax (section 4).
The temperature is then fitted on held-out OUTCOMES, never on teacher labels: the goal
is to be calibrated against the world, not against the teacher.
"""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .backend_impls.transformers import best_device
from .calibration.harness import Prediction, diagnose_temperature
from .calibration.temperature import sharpen
from .gate import GatePolicy, compare
from .prompt import declared_options, state_body
from .scoring import choice_from_probs, noul_from_probs, score_from_probs
from .traffic import TEACHER_SOURCE, Example, family_key, question_to_dict
from .types import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion, question_from_dict

DEFAULT_STUDENT = "answerdotai/ModernBERT-base"


def state_text(state: object) -> str:
    """Same serialisation the teacher's prompt uses for the state body."""
    return state_body(state)


def targets(example: Example, mix: float) -> list[float]:
    if example.source != TEACHER_SOURCE:
        # A head answered this one: its distribution is the student's own opinion and
        # must never be a target. Only the outcome may be learned from.
        if example.outcome is None:
            raise ValueError("a head-served example is trainable only with an outcome")
        return [1.0 if i == example.outcome else 0.0 for i in range(len(example.teacher))]
    if example.outcome is None:
        return list(example.teacher)
    hot = [1.0 if i == example.outcome else 0.0 for i in range(len(example.teacher))]
    return [mix * h + (1.0 - mix) * t for h, t in zip(hot, example.teacher)]


def trainable(examples: Sequence[Example]) -> list[Example]:
    """Teacher-answered rows, plus head-answered rows that have an outcome."""
    return [e for e in examples if e.source == TEACHER_SOURCE or e.outcome is not None]


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 5e-5
    batch_size: int = 16
    max_length: int = 256
    outcome_mix: float = 1.0      # 1.0: a known outcome fully replaces the teacher
    seed: int = 0


class Head:
    """A trained student for one question family."""

    def __init__(self, model, tokenizer, question: Question, device: str,
                 temperature: float = 1.0, max_length: int = 256, base: str = "",
                 evaluation: dict | None = None, version: str | None = None):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.question = question
        self.family = family_key(question)
        self.device = device
        self.temperature = temperature
        self.max_length = max_length
        self.base = base
        # Written by the promotion gate (see main()); the router serves only heads
        # whose evaluation says promoted.
        self.evaluation = evaluation or {}
        # Identifies these exact weights, so shadow answers are only ever credited to
        # the head that produced them -- not to a retrained successor.
        self.version = version or uuid.uuid4().hex[:12]

    @property
    def label(self) -> str:
        return f"head:{self.family}"

    @property
    def promoted(self) -> bool:
        return bool(self.evaluation.get("promoted"))

    @property
    def options(self) -> list[str]:
        return declared_options(self.question)

    @torch.inference_mode()
    def predict_proba(self, states: Sequence[object], batch_size: int = 32,
                      temperature: float | None = None) -> list[list[float]]:
        t = self.temperature if temperature is None else temperature
        out = []
        for i in range(0, len(states), batch_size):
            batch = self.tokenizer([state_text(s) for s in states[i:i + batch_size]],
                                   padding=True, truncation=True,
                                   max_length=self.max_length, return_tensors="pt").to(self.device)
            probs = torch.softmax(self.model(**batch).logits.float(), dim=-1).tolist()
            out.extend(sharpen(p, t) for p in probs)
        return out

    def answer(self, state: object, confidence: str = "margin"):
        """A typed answer, same shape Decider.ask() returns for this question."""
        return self.to_answer(self.predict_proba([state])[0], confidence)

    def to_answer(self, probs: Sequence[float], confidence: str = "margin"):
        probs = list(probs)
        if isinstance(self.question, ChoiceQuestion):
            return choice_from_probs(probs, self.question.criteria, confidence)
        if isinstance(self.question, NoulQuestion):
            return noul_from_probs(probs[0])
        return score_from_probs(probs, self.question.criteria, confidence)

    def fit_temperature(self, labelled: Sequence[Example]):
        """Fit on held-out examples WITH outcomes; returns the harness's TemperatureFit.

        An untrustworthy fit (separable or at a search bound) is reported but NOT
        applied: the head keeps T=1. A small, all-correct calibration set otherwise
        drives T toward 0 and makes the head certain of everything.
        """
        rows = [e for e in labelled if e.outcome is not None]
        if not rows:
            raise ValueError("temperature needs held-out examples with outcomes")
        probs = self.predict_proba([e.state for e in rows], temperature=1.0)
        fit = diagnose_temperature([
            Prediction(dict(zip(self.options, p)), self.options[e.outcome])
            for p, e in zip(probs, rows)
        ])
        self.temperature = fit.temperature if fit.trustworthy else 1.0
        return fit

    def save(self, path: str | Path) -> None:
        path = Path(path)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / "head.json").write_text(json.dumps({
            "question": question_to_dict(self.question),
            "family": self.family,
            "temperature": self.temperature,
            "max_length": self.max_length,
            "base": self.base,
            "evaluation": self.evaluation,
            "version": self.version,
        }, indent=2))

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "Head":
        path = Path(path)
        meta = json.loads((path / "head.json").read_text())
        question = question_from_dict(meta["question"])
        if family_key(question) != meta["family"]:
            raise ValueError("head metadata is inconsistent: family does not match question")
        return cls(AutoModelForSequenceClassification.from_pretrained(path),
                   AutoTokenizer.from_pretrained(path), question, best_device(device),
                   meta["temperature"], meta["max_length"], meta.get("base", ""),
                   meta.get("evaluation"), meta.get("version"))


def train_head(examples: Sequence[Example], base: str = DEFAULT_STUDENT,
               config: TrainConfig | None = None, device: str | None = None,
               log=print) -> Head:
    config = config or TrainConfig()
    if not examples:
        raise ValueError("nothing to train on")
    families = {e.family for e in examples}
    if len(families) != 1:
        raise ValueError(f"a head is one family; got {len(families)}. Filter with "
                         "traffic.examples(log, family=...)")
    question = examples[0].question
    n = len(declared_options(question))

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = best_device(device)
    tokenizer = AutoTokenizer.from_pretrained(base)
    # A fresh head per family: a base that already has a classifier of another size
    # (e.g. a previous head) must have it replaced, not refuse to load.
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=n, ignore_mismatched_sizes=True).to(device)
    model.train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)

    rows = [(state_text(e.state), targets(e, config.outcome_mix)) for e in trainable(examples)]
    if not rows:
        raise ValueError("no trainable rows: head-served examples need outcomes")
    steps = config.epochs * ((len(rows) + config.batch_size - 1) // config.batch_size)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: max(0.0, 1.0 - step / max(steps, 1)))

    for epoch in range(config.epochs):
        random.shuffle(rows)
        total = 0.0
        for i in range(0, len(rows), config.batch_size):
            texts, soft = zip(*rows[i:i + config.batch_size])
            batch = tokenizer(list(texts), padding=True, truncation=True,
                              max_length=config.max_length, return_tensors="pt").to(device)
            target = torch.tensor(soft, dtype=torch.float32, device=device)
            logits = model(**batch).logits.float()
            loss = -(target * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            schedule.step()
            optimiser.zero_grad()
            total += loss.item() * len(texts)
        log(f"  epoch {epoch + 1}/{config.epochs}  log-loss {total / len(rows):.4f}")

    return Head(model, tokenizer, question, device, max_length=config.max_length, base=base)


# --- promotion gate ---------------------------------------------------------------

def _bucket(request_id: str) -> float:
    """Stable pseudo-random position in [0, 1): the split survives re-runs and log growth."""
    import hashlib
    return int(hashlib.sha256(request_id.encode()).hexdigest()[:8], 16) / 0x100000000


def split(rows: Sequence[Example], holdout: float):
    """(train, temperature, evaluation). Only teacher-answered rows with outcomes can be
    held out -- the teacher's logged distribution is the baseline being compared against."""
    held = [e for e in rows if e.outcome is not None and e.source == TEACHER_SOURCE
            and _bucket(e.request_id) < holdout]
    held_ids = {e.request_id for e in held}
    train = [e for e in trainable(rows) if e.request_id not in held_ids]
    temperature = [e for e in held if _bucket(e.request_id) < holdout / 2]
    evaluation = [e for e in held if _bucket(e.request_id) >= holdout / 2]
    return train, temperature, evaluation


def promote(head: Head, evaluation: Sequence[Example], policy: GatePolicy = GatePolicy(),
            student: Sequence[Sequence[float]] | None = None, source: str = "holdout") -> dict:
    """Gate the head against the teacher on outcomes it never trained on (see gate.py).

    `student` defaults to the head's current predictions; shadow evaluation passes the
    distributions it actually logged at serving time instead.
    """
    rows = [e for e in evaluation if e.outcome is not None]
    if student is None:
        student = head.predict_proba([e.state for e in rows]) if rows else []
    verdict = compare(student, [e.teacher for e in rows], [e.outcome for e in rows], policy)
    verdict.update(source=source, temperature=head.temperature, version=head.version)
    return verdict


def _policy_args(parser) -> None:
    d = GatePolicy()
    parser.add_argument("--min-eval", type=int, default=d.min_eval)
    parser.add_argument("--brier-margin", type=float, default=d.brier_margin)
    parser.add_argument("--max-accuracy-drop", type=float, default=d.max_accuracy_drop)
    parser.add_argument("--confidence", type=float, default=d.confidence)


def _policy(args) -> GatePolicy:
    return GatePolicy(min_eval=args.min_eval, brier_margin=args.brier_margin,
                      max_accuracy_drop=args.max_accuracy_drop, confidence=args.confidence)


def shadow_rows(rows: Sequence[Example], head: Head) -> tuple[list[Example], list[list[float]]]:
    """Teacher-served rows with an outcome where THIS head version ran in shadow."""
    picked = [e for e in rows if e.outcome is not None and e.source == TEACHER_SOURCE
              and e.shadow and e.shadow.get("head") == head.label
              and e.shadow.get("version") == head.version]
    return picked, [list(e.shadow["probabilities"]) for e in picked]


def _shadow(args) -> None:
    from .traffic import TrafficLog, examples
    head = Head.load(args.head, device="cpu")
    rows, dists = shadow_rows(examples(TrafficLog(args.log), family=head.family), head)
    verdict = promote(head, rows, _policy(args), student=dists, source="shadow")
    print(json.dumps(verdict, indent=2))
    if args.write:
        meta_path = Path(args.head) / "head.json"
        meta = json.loads(meta_path.read_text())
        meta["evaluation"] = verdict
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"wrote verdict to {meta_path} ({'PROMOTED' if verdict['promoted'] else 'not promoted'})")


def main(argv=None) -> None:
    import argparse
    from collections import Counter
    from .traffic import TrafficLog, examples

    parser = argparse.ArgumentParser(prog="python -m typed_decisions.distill")
    sub = parser.add_subparsers(dest="command", required=True)
    fam = sub.add_parser("families", help="list question families in a traffic log")
    fam.add_argument("--log", required=True)
    tr = sub.add_parser("train", help="train, calibrate and gate one family's head")
    tr.add_argument("--log", required=True)
    tr.add_argument("--family", required=True)
    tr.add_argument("--out", required=True, help="heads directory; writes <out>/<family>/")
    tr.add_argument("--base", default=DEFAULT_STUDENT)
    tr.add_argument("--epochs", type=int, default=10)
    tr.add_argument("--holdout", type=float, default=0.3)
    _policy_args(tr)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--device")
    sh = sub.add_parser("shadow", help="gate a head on the answers it gave in shadow mode")
    sh.add_argument("--log", required=True)
    sh.add_argument("--head", required=True, help="the head's directory")
    sh.add_argument("--write", action="store_true",
                    help="store the verdict in the head, so --heads serves it if promoted")
    _policy_args(sh)
    args = parser.parse_args(argv)

    if args.command == "shadow":
        return _shadow(args)

    rows = examples(TrafficLog(args.log), family=None if args.command == "families" else args.family)
    if args.command == "families":
        count = Counter(e.family for e in rows)
        labelled = Counter(e.family for e in rows if e.outcome is not None)
        seen = {}
        for e in rows:
            seen.setdefault(e.family, e.question)
        for key, question in seen.items():
            print(f"{key}  asks={count[key]:<6} outcomes={labelled[key]:<6} "
                  f"{question.type}: {question.instructions}")
        return

    if not rows:
        raise SystemExit(f"no examples for family {args.family}")
    train, temp, evaluation = split(rows, args.holdout)
    print(f"train={len(train)} temperature={len(temp)} eval={len(evaluation)}")
    head = train_head(train, base=args.base, device=args.device,
                      config=TrainConfig(epochs=args.epochs, seed=args.seed))
    if temp:
        print(f"temperature: {head.fit_temperature(temp).summary()}")
    head.evaluation = promote(head, evaluation, _policy(args))
    out = Path(args.out) / head.family
    head.save(out)
    print(json.dumps(head.evaluation, indent=2))
    print(f"wrote {out}  ({'PROMOTED' if head.promoted else 'not promoted'})")


if __name__ == "__main__":
    main()
