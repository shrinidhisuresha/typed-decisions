"""Phase 3 pilot: distil the debiased 4B teacher into a ModernBERT head, end to end.

    python examples/distill_pilot.py [--teacher-cache basecal_cache.json] [--out docs/x.md]

Walks the real pipeline: teacher answers are written to a TrafficLog, outcomes are
recorded against request ids, traffic.examples() joins them, train_head() fits the
student, fit_temperature() calibrates it on held-out outcomes, and the harness reports
on a test split neither the student nor its temperature ever saw.

This validates the PIPELINE. 60 training tickets is two orders of magnitude short of
the "few thousand per family" the design doc asks for; do not read the student's
accuracy as what distillation can do.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import tempfile
import time
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(__file__))
from calibrate import DATASET, QUESTION, ROUTES, stratified_split  # noqa: E402

from typed_decisions import Calibration, Decider, Prediction, diagnose_temperature, report  # noqa: E402
from typed_decisions.backend_impls.transformers import TransformersBackend  # noqa: E402
from typed_decisions.distill import DEFAULT_STUDENT, TrainConfig, train_head  # noqa: E402
from typed_decisions.traffic import TrafficLog, examples, family_key  # noqa: E402

TEACHER = "Qwen/Qwen3.5-4B"
PERMUTATIONS = 4


def teacher_client():
    backend = TransformersBackend.from_pretrained(TEACHER, dtype=torch.bfloat16)
    return Decider(backend, calibration=Calibration(permutations=PERMUTATIONS, contextual=True))


def teacher_distributions(cache_path: str | None) -> tuple[list[list[float]], object]:
    if cache_path and os.path.exists(cache_path):
        cache = json.load(open(cache_path))
        if TEACHER in cache:
            return cache[TEACHER], None
    client = teacher_client()
    out = []
    for text, _ in DATASET:
        probs = client.ask({"ticket": text}, QUESTION).answers["route"].probabilities
        out.append([probs[r] for r in ROUTES])
    return out, client


def median_ms(fn, repeats: int) -> float:
    fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def evaluate(name: str, preds: list[Prediction], temperature: float) -> str:
    rep = report(preds, temperature=temperature)
    margin = report(preds, temperature=temperature, confidence="margin")
    hist = Counter(max(p.probabilities, key=p.probabilities.get) for p in preds)
    cov = [pt for pt in margin.selective if pt.threshold == 0.5]
    at50 = (f"{cov[0].coverage:.2f} / {cov[0].accuracy:.3f}"
            if cov and cov[0].accuracy is not None else "n/a")
    return (f"| {name} | {temperature:.2f} | {rep.accuracy:.3f} | {rep.ece:.3f} | "
            f"{rep.brier:.3f} | {at50} | "
            + ", ".join(f"{r.split()[0]} {hist.get(r, 0)}" for r in ROUTES) + " |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-cache")
    parser.add_argument("--student", default=DEFAULT_STUDENT)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--out", default="distill_pilot.md")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-latency", action="store_true")
    args = parser.parse_args()

    teacher, client = teacher_distributions(args.teacher_cache)
    by_text = {text: probs for (text, _), probs in zip(DATASET, teacher)}
    train, test = stratified_split(DATASET)
    fit_rows, temp_rows = [r for i, r in enumerate(train) if i % 4], train[::4]

    # 1. Traffic: every train ticket is answered by the teacher and logged; its outcome
    #    arrives later. Test tickets are never logged -- the student must not see them.
    log = TrafficLog(os.path.join(tempfile.mkdtemp(), "traffic.jsonl"))
    for text, label in train:
        answer = {"route": {"choice": None, "confidence": 0.0,
                            "probabilities": dict(zip(ROUTES, by_text[text]))}}
        rid = log.record_ask({"ticket": text}, QUESTION, {"answers": answer}, TEACHER)
        log.record_outcome(rid, {"route": label})

    # 2. Join, then split the logged traffic into fit / temperature sets.
    family = family_key(QUESTION["route"])
    rows = {e.state["ticket"]: e for e in examples(log, family=family)}
    fit_examples = [rows[t] for t, _ in fit_rows]
    temp_examples = [rows[t] for t, _ in temp_rows]

    # Teacher reference: its temperature fitted on the same held-out rows.
    teacher_t = diagnose_temperature(
        [Prediction(dict(zip(ROUTES, by_text[t])), l) for t, l in temp_rows]).temperature
    teacher_test = [Prediction(dict(zip(ROUTES, by_text[t])), l) for t, l in test]

    lines = [
        f"# Distillation pilot: {TEACHER} -> {args.student}",
        "",
        f"Teacher: permutations={PERMUTATIONS} + contextual. Student trained on "
        f"{len(fit_examples)} logged tickets, temperature fitted on {len(temp_examples)} "
        f"held-out logged tickets, reported on {len(test)} test tickets never logged. "
        f"{args.epochs} epochs, soft cross-entropy (log score), seed {args.seed}.",
        "",
        "| model | T | acc | ECE | Brier | margin>=0.5: cov / acc | predicted classes |",
        "|---|---:|---:|---:|---:|---:|---|",
        evaluate("teacher (4B, debiased)", teacher_test, teacher_t),
    ]
    print(lines[-1], flush=True)

    heads = {}
    for name, mix in (("student, teacher labels only", 0.0), ("student, outcomes", 1.0)):
        print(f"training: {name}", flush=True)
        head = train_head(fit_examples, base=args.student,
                          config=TrainConfig(epochs=args.epochs, outcome_mix=mix, seed=args.seed))
        fit = head.fit_temperature(temp_examples)
        probs = head.predict_proba([{"ticket": t} for t, _ in test], temperature=1.0)
        preds = [Prediction(dict(zip(ROUTES, p)), l) for p, (_, l) in zip(probs, test)]
        lines.append(evaluate(name, preds, fit.temperature))
        print(lines[-1], flush=True)
        heads[name] = head

    if not args.no_latency:
        state = {"ticket": test[0][0]}
        head = heads["student, outcomes"]
        student_ms = median_ms(lambda: head.predict_proba([state]), 20)
        del heads
        gc.collect()
        client = client or teacher_client()
        teacher_ms = median_ms(lambda: client.ask(state, QUESTION), 5)
        lines += ["", "Latency for one ticket, one question, median, Apple M5 Pro (MPS):", "",
                  "| model | ms | |", "|---|---:|---:|",
                  f"| teacher 4B, {PERMUTATIONS} permutations + contextual | {teacher_ms:.0f} | 1x |",
                  f"| student {args.student.split('/')[-1]} | {student_ms:.1f} | "
                  f"{teacher_ms / student_ms:.0f}x |"]
        print(lines[-3:], flush=True)

    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
