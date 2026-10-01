"""typed-decisions on the public benchmarks: all three primitives, real labels.

    python examples/benchmark.py [--model Qwen/Qwen3.5-4B] [--tasks all] [--per-class 20]

For each task in typed_decisions.benchmarks: sample balanced rows, split in half per class, score
both halves, fit a temperature on train, report on test. Metrics are computed from the
full distribution over the task's options, so Choice, Score and Noul are measured the same
way (accuracy of the argmax, ECE on max-probability, multi-class Brier), plus the error
of the expected score for Score tasks.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections import Counter

import torch

from typed_decisions import Calibration, Decider, Prediction, diagnose_temperature, report
from typed_decisions.backend_impls.transformers import TransformersBackend
from typed_decisions.benchmarks import TASKS, load, split
from typed_decisions.calibration.harness import _apply
from typed_decisions.types import NoulQuestion, ScoreQuestion


def distribution(task, answer) -> dict[str, float]:
    if isinstance(task.question, NoulQuestion):
        return {"yes": answer.noul, "no": 1.0 - answer.noul}
    return dict(answer.probabilities)


def run(client, task, rows):
    preds, ms = [], []
    for row in rows:
        start = time.perf_counter()
        answer = client.ask(row.state, {"q": task.question}).answers["q"]
        ms.append((time.perf_counter() - start) * 1e3)
        preds.append(Prediction(distribution(task, answer), row.label))
    return preds, statistics.median(ms)


def score_mae(task, preds) -> float | None:
    if not isinstance(task.question, ScoreQuestion):
        return None
    legend = task.question.criteria
    return statistics.mean(abs(sum(legend[k] * v for k, v in p.probabilities.items())
                               - legend[p.label]) for p in preds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--permutations", type=int, default=1)
    parser.add_argument("--no-contextual", action="store_true")
    parser.add_argument("--retrieve", type=int, default=0,
                        help="shortlist size: embed-and-shortlist Choices over 26 options")
    parser.add_argument("--out", default="benchmark.md")
    args = parser.parse_args()
    names = list(TASKS) if args.tasks == "all" else args.tasks.split(",")

    backend = TransformersBackend.from_pretrained(args.model, dtype=torch.bfloat16)
    retriever = None
    if args.retrieve:
        from typed_decisions.retrieval import EmbeddingRetriever
        retriever = EmbeddingRetriever()
    client = Decider(backend, calibration=Calibration(permutations=args.permutations,
                                                  contextual=not args.no_contextual),
                 retriever=retriever, shortlist_k=max(args.retrieve, 2))
    lines = [
        f"# Public benchmarks: {args.model}, bfloat16 on MPS",
        "",
        f"Balanced samples, {args.per_class} per class, split in half per class; "
        f"permutations={args.permutations}, contextual "
        f"{'off' if args.no_contextual else 'on'}"
        + (f", retrieval top-{args.retrieve} above 26 options" if args.retrieve else "")
        + "; temperature fitted on train, all metrics "
        "on test. Datasets are synthetic with noisy labels: compare systems, don't read "
        "absolute accuracy as a ceiling.",
        "",
        "| task | type | options | n test | chance | T | acc | ECE | Brier | score MAE "
        "| ms p50 | most predicted |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in names:
        task = TASKS[name]
        train, test = split(load(name, args.per_class))
        print(f"{name}: {len(train)} train / {len(test)} test", flush=True)
        train_preds, _ = run(client, task, train)
        test_preds, ms = run(client, task, test)
        fit = diagnose_temperature(train_preds)
        t = fit.temperature if fit.trustworthy else 1.0
        scaled = _apply(test_preds, t)
        rep = report(scaled)
        mae = score_mae(task, scaled)
        top = Counter(max(p.probabilities, key=p.probabilities.get) for p in scaled)
        lines.append(
            f"| {name} | {task.question.type} | {len(task.labels)} | {len(test)} | "
            f"{1 / len(task.labels):.3f} | {t:.2f} | {rep.accuracy:.3f} | {rep.ece:.3f} | "
            f"{rep.brier:.3f} | {'' if mae is None else f'{mae:.3f}'} | {ms:.0f} | "
            + ", ".join(f"{k} {v}" for k, v in top.most_common(3)) + " |")
        print(lines[-1], flush=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
