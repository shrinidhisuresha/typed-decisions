"""Latency per decision, per stage and end to end, in ms and µs (p50 / p95 / p99).

    python examples/latency.py [--runs 200] [--out docs/latency.md]

Paths timed on BANKING77 queries (77 options, real user text), one query at a time after a
warm-up, with time.perf_counter_ns and a device sync before each clock read:

  probe      embed the query + a linear layer over 77 classes; MiniLM-L6 (23M) on the CPU,
             and Qwen3-Embedding-0.6B on the Apple GPU
  head       ModernBERT-base classification head (forward pass only)
  zero-shot  the full Decider: Qwen3.5-9B 4-bit (MLX), retrieval, 1 or 4 orderings,
             contextual prior

The probe's linear layer is also timed on its own. It runs in microseconds, but it is a
component: never quote it as the end-to-end number.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from typed_decisions import Calibration, Decider
from typed_decisions.benchmarks import TASKS, load, split
from typed_decisions.prompt import state_body


def sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, inputs, runs, warmup=5):
    for x in inputs[:warmup]:
        fn(x)
    sync()
    samples = []
    for i in range(runs):
        x = inputs[i % len(inputs)]
        start = time.perf_counter_ns()
        fn(x)
        sync()
        samples.append(time.perf_counter_ns() - start)
    return samples


def pct(samples, q):
    s = sorted(samples)
    return s[min(len(s) - 1, int(q * len(s)))]


def row(name, ns):
    ms = lambda v: v / 1e6
    return (f"| {name} | {ms(pct(ns, .5)):.2f} | {ms(pct(ns, .95)):.2f} | {ms(pct(ns, .99)):.2f} "
            f"| {pct(ns, .5) / 1e3:,.0f} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--llm-runs", type=int, default=30)
    parser.add_argument("--out", default="latency.md")
    args = parser.parse_args()

    task = TASKS["banking77"]
    train, test = split(load("banking77", 10))
    texts = [r.state for r in test]
    lines = ["# Latency per decision (BANKING77, 77 options, one query at a time)", "",
             "| path | p50 ms | p95 ms | p99 ms | p50 µs |", "|---|---:|---:|---:|---:|"]

    # fast probe: 23M MiniLM on the CPU (the speed sweep's pick)
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from benchmark_labels import Embedder
    mini = Embedder("sentence-transformers/all-MiniLM-L6-v2", "mean", "", device="cpu")
    Wm = torch.randn(mini([texts[0]]).shape[1], len(task.labels))
    lines.append(row("**probe end to end, MiniLM-L6 23M on CPU**",
                     timed(lambda t: torch.softmax(mini([t]) @ Wm, dim=1), texts, args.runs)))

    # probe: embedding + linear
    from typed_decisions.retrieval import EmbeddingRetriever
    r = EmbeddingRetriever()
    X = r._embed([state_body(s.state) for s in train])
    W = torch.randn(X.shape[1], len(task.labels))          # weights' values don't affect timing
    embed = lambda t: r._embed([state_body(t)])
    lines.append(row("probe: embed query (0.6B)", timed(embed, texts, args.runs)))
    v = embed(texts[0])
    lines.append(row("probe: linear layer only (component)",
                     timed(lambda _: torch.softmax(v @ W, dim=1), texts, args.runs)))
    lines.append(row("probe end to end, Qwen3-Embedding-0.6B on the Apple GPU",
                     timed(lambda t: torch.softmax(embed(t) @ W, dim=1), texts, args.runs)))

    # ModernBERT head forward pass
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from typed_decisions.backend_impls.transformers import best_device
    device = best_device(None)
    tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    head = AutoModelForSequenceClassification.from_pretrained(
        "answerdotai/ModernBERT-base", num_labels=len(task.labels)).to(device).eval()

    @torch.inference_mode()
    def head_fn(t):
        batch = tok([t], return_tensors="pt", truncation=True, max_length=256).to(device)
        return torch.softmax(head(**batch).logits, dim=-1)
    lines.append(row("**ModernBERT-base head end to end**", timed(head_fn, texts, args.runs)))
    del head

    # zero-shot decoder path
    from typed_decisions.backend_impls.mlx import MLXBackend
    backend = MLXBackend.from_pretrained("mlx-community/Qwen3.5-9B-4bit")
    for perms in (4, 1):
        d = Decider(backend, calibration=Calibration(permutations=perms, contextual=True),
                    retriever=r, shortlist_k=10)
        q = {"q": task.question}
        lines.append(row(f"zero-shot 9B 4-bit (MLX), {perms} ordering(s), retrieval, contextual",
                         timed(lambda t: d.ask(t, q), texts, args.llm_runs, warmup=2)))

    lines += ["", f"Apple M5 Pro, {device}; {args.runs} timed runs ({args.llm_runs} for the 9B "
              "rows) after warm-up; perf_counter_ns with a device sync. The linear layer alone is "
              "a component figure, not a decision latency."]
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
