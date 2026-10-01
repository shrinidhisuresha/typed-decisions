"""BaseCal on your data (design doc section 9): does pooling in the -Base twin help?

Scores the calibrate.py ticket set with the instruct model and its -Base twin (both
permutation + contextual debiased), then for each pooling weight w fits a temperature
on train and reports on held-out test. w=0 is the instruct-only Phase 1 baseline.

    python examples/basecal.py [instruct] [base] [--out docs/basecal.md]

Models are loaded one at a time and their distributions cached to --cache, so a 4B
pair fits a 24 GB Mac and re-running the analysis costs nothing.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(__file__))
from calibrate import DATASET, QUESTION, ROUTES, stratified_split  # noqa: E402

from typed_decisions import Calibration, Decider, Prediction, diagnose_temperature, report  # noqa: E402
from typed_decisions.backend_impls.transformers import TransformersBackend  # noqa: E402
from typed_decisions.calibration.basecal import pool  # noqa: E402
from typed_decisions.scoring import confidence_from  # noqa: E402

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
PERMUTATIONS = 4


def distributions(model: str, rows, device: str | None) -> list[list[float]]:
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    backend = TransformersBackend.from_pretrained(model, device=device, dtype=dtype)
    client = Decider(backend, calibration=Calibration(permutations=PERMUTATIONS, contextual=True))
    out = []
    for i, (text, _) in enumerate(rows, 1):
        probs = client.ask({"ticket": text}, QUESTION).answers["route"].probabilities
        out.append([probs[r] for r in ROUTES])
        if i % 20 == 0:
            print(f"  {model}: {i}/{len(rows)}", flush=True)
    del client, backend
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return out


def cached(model: str, rows, cache: dict, device) -> list[list[float]]:
    if model not in cache:
        cache[model] = distributions(model, rows, device)
    return cache[model]


def accuracy_at_coverage(preds: list[Prediction], coverage: float) -> float:
    """Keep the most confident `coverage` fraction by margin; how often are they right?"""
    scored = sorted(
        ((confidence_from(list(p.probabilities.values()), "margin"),
          max(p.probabilities, key=p.probabilities.get) == p.label) for p in preds),
        reverse=True,
    )
    kept = scored[: max(1, round(len(scored) * coverage))]
    return sum(ok for _, ok in kept) / len(kept)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("instruct", nargs="?", default="Qwen/Qwen3.5-4B")
    parser.add_argument("base", nargs="?", default="Qwen/Qwen3.5-4B-Base")
    parser.add_argument("--device")
    parser.add_argument("--cache", default="basecal_cache.json")
    parser.add_argument("--out", default="basecal.md")
    args = parser.parse_args()

    rows = DATASET  # scored once; split afterwards so both halves share the cache
    cache = json.load(open(args.cache)) if os.path.exists(args.cache) else {}
    instruct = cached(args.instruct, rows, cache, args.device)
    json.dump(cache, open(args.cache, "w"))
    base = cached(args.base, rows, cache, args.device)
    json.dump(cache, open(args.cache, "w"))

    index = {row: i for i, row in enumerate(rows)}
    train, test = stratified_split(rows)

    lines = [
        f"# BaseCal: {args.instruct} + {args.base}",
        "",
        f"{len(train)} train / {len(test)} test tickets, 4 routes, both models debiased "
        f"(permutations={PERMUTATIONS} + contextual). Pool: log p = (1-w) log p_instruct + "
        "w log p_base; temperature fitted on train per row, all metrics on test.",
        "",
        "| w | T | acc | ECE | margin ECE | Brier | acc @50% cov | acc @25% cov | predicted classes |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for w in WEIGHTS:
        def preds(split):
            return [Prediction(dict(zip(ROUTES, pool(instruct[index[r]], base[index[r]], w))), r[1])
                    for r in split]
        fit = diagnose_temperature(preds(train))
        t = fit.temperature
        test_preds = preds(test)
        rep = report(test_preds, temperature=t)
        margin = report(test_preds, temperature=t, confidence="margin")
        scaled = [Prediction(p.probabilities, p.label) for p in test_preds]
        hist = Counter(max(p.probabilities, key=p.probabilities.get) for p in scaled)
        hist_s = ", ".join(f"{r.split()[0]} {hist.get(r, 0)}" for r in ROUTES)
        flag = " (at bound)" if fit.at_bound else ""
        lines.append(
            f"| {w:.2f} | {t:.2f}{flag} | {rep.accuracy:.3f} | {rep.ece:.3f} | {margin.ece:.3f} | "
            f"{rep.brier:.3f} | {accuracy_at_coverage(scaled, 0.5):.3f} | "
            f"{accuracy_at_coverage(scaled, 0.25):.3f} | {hist_s} |")
        print(lines[-1], flush=True)

    lines += ["", f"Test labels: {dict(Counter(label for _, label in test))}"]
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
