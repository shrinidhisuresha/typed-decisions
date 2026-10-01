"""Sequence scoring (Choice beyond 26 options) measured on the labelled ticket set.

    python examples/high_cardinality.py [model] [--out docs/x.md] [--permutations 4 --configs AB]

A  letter scoring, 4 routes       -- the Phase 1 method, the baseline
B  sequence scoring, same 4 routes -- isolates the scoring method itself
C  sequence scoring, 40 options    -- the 4 routes among 36 unrelated departments
D  retrieve top-k, then letters    -- same 40 options, embedding shortlist first (3C)

Labels are still the 4 true routes, so picking a distractor is simply wrong. Every row
uses contextual calibration and a temperature fitted on the train split. Default
permutations=1, because C with k rotations costs k x 40 option forks per ticket.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(__file__))
from calibrate import DATASET, ROUTES, stratified_split  # noqa: E402

from typed_decisions import Calibration, Decider, Prediction, diagnose_temperature, report  # noqa: E402
from typed_decisions.backend_impls.transformers import TransformersBackend  # noqa: E402
from typed_decisions.types import ChoiceQuestion  # noqa: E402

DISTRACTORS = [
    "legal", "human resources", "facilities", "marketing", "public relations", "recruiting",
    "office management", "travel desk", "catering", "shipping and logistics", "warehouse",
    "quality assurance", "research and development", "product design", "internal audit",
    "investor relations", "corporate communications", "events", "training and development",
    "employee wellness", "real estate", "fleet management", "procurement",
    "translation services", "content moderation", "community management", "social media",
    "brand design", "localization", "partnerships", "government affairs", "sustainability",
    "diversity and inclusion", "mailroom", "reception", "security operations",
]
INSTRUCTIONS = "Which team should handle this ticket?"


def run(client, question, rows):
    preds, ms = [], []
    for text, label in rows:
        start = time.perf_counter()
        probs = client.ask({"ticket": text}, {"route": question}).answers["route"].probabilities
        ms.append((time.perf_counter() - start) * 1e3)
        preds.append(Prediction(probs, label))
    return preds, statistics.median(ms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--out", default="high_cardinality.md")
    parser.add_argument("--permutations", type=int, default=1)
    parser.add_argument("--configs", default="ABC", help="which of A, B, C, D to run")
    parser.add_argument("--k", type=int, default=10, help="D: shortlist size")
    parser.add_argument("--no-contextual", action="store_true")
    parser.add_argument("--sequence-norm", choices=["mean", "sum"], default="mean",
                        help="sum + contextual = PMI scoring of option text")
    args = parser.parse_args()
    assert len(DISTRACTORS) == 36 and not set(DISTRACTORS) & set(ROUTES)

    backend = TransformersBackend.from_pretrained(args.model, dtype=torch.bfloat16)
    cal = Calibration(contextual=not args.no_contextual, permutations=args.permutations)
    many = ChoiceQuestion(INSTRUCTIONS, ROUTES + DISTRACTORS)
    configs = [
        ("A letter, 4 options", Decider(backend, calibration=cal), ChoiceQuestion(INSTRUCTIONS, ROUTES)),
        ("B sequence, 4 options", Decider(backend, calibration=cal, sequence_threshold=0,
                                      sequence_norm=args.sequence_norm),
         ChoiceQuestion(INSTRUCTIONS, ROUTES)),
        ("C sequence, 40 options", Decider(backend, calibration=cal,
                                       sequence_norm=args.sequence_norm), many),
    ]
    retriever = None
    if "D" in args.configs:
        from typed_decisions.retrieval import EmbeddingRetriever
        retriever = EmbeddingRetriever()
        configs.append((f"D retrieve top-{args.k}, then letters, 40 options",
                        Decider(backend, calibration=cal, retriever=retriever, shortlist_k=args.k),
                        many))
    configs = [c for c in configs if c[0][0] in args.configs]
    train, test = stratified_split(DATASET)
    lines = [
        f"# Choice beyond 26 options: {args.model}, bfloat16 on MPS",
        "",
        f"{len(train)} train / {len(test)} test tickets, "
        f"contextual calibration {'off' if args.no_contextual else 'on'}, "
        f"sequence scores: {args.sequence_norm} log-prob, "
        f"permutations={args.permutations}, "
        "temperature fitted on train. C adds 36 unrelated departments as options; labels are "
        "the 4 true routes.",
        "",
        "| config | T | acc | ECE | Brier | ms / ticket (p50) | top predictions |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, client, question in configs:
        print(f"{name} ...", flush=True)
        train_preds, _ = run(client, question, train)
        test_preds, ms = run(client, question, test)
        fit = diagnose_temperature(train_preds)
        t = fit.temperature if fit.trustworthy else 1.0
        rep = report(test_preds, temperature=t)
        top = Counter(max(p.probabilities, key=p.probabilities.get) for p in test_preds)
        shown = ", ".join(f"{k} {v}" for k, v in top.most_common(5))
        if name.startswith("D"):
            from typed_decisions.prompt import state_body
            from typed_decisions.retrieval import shortlist
            hits = 0
            for text, label in test:
                scores = retriever.rank(state_body({"ticket": text}), many.criteria,
                                        task=f"Find the answer option that best answers: "
                                        f"{many.instructions}")
                hits += many.criteria.index(label) in shortlist(scores, args.k)
            shown = f"recall@{args.k} {hits / len(test):.3f}; " + shown
        lines.append(f"| {name} | {t:.2f} | {rep.accuracy:.3f} | {rep.ece:.3f} | "
                     f"{rep.brier:.3f} | {ms:.0f} | {shown} |")
        print(lines[-1], flush=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
