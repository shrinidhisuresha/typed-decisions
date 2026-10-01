"""Phase 2 measurement (design doc section 7): cost and latency per question at k=1 vs k=20,
shared-prefix fork vs. naive re-prefill, plus the prefix cache hit rate.

    python examples/bench_prefix.py [model] [--out bench.md] [--repeats 3]

Runs on TransformersBackend, where both paths exist side by side: score() prefills the
state once and forks; score_uncached() re-reads the whole prompt per question, which is
what a server without prefix caching (or a template that puts the question first) does.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from typed_decisions.backend_impls.transformers import TransformersBackend
from typed_decisions.client import Decider
from typed_decisions.prompt import render_prefix
from typed_decisions.types import NoulQuestion

TICKET = ("Customer reports intermittent 502 errors on the checkout API since the last "
          "deploy; retries sometimes succeed. They are on the enterprise plan. ")


def state_of(approx_tokens: int) -> dict:
    # ~25 tokens per sentence; history is what makes a real decision state long.
    return {"ticket": TICKET, "history": [TICKET] * max(approx_tokens // 25, 1)}


def questions(k: int) -> dict:
    return {f"q{i}": NoulQuestion(f"Question {i}: does this ticket need escalation tier {i}?")
            for i in range(k)}


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def timed(fn, device: str, repeats: int) -> float:
    fn()  # warm-up: kernels, allocator
    sync(device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        sync(device)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device")
    parser.add_argument("--out", default="bench_prefix.md")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--states", default="200,2000")
    args = parser.parse_args()

    backend = TransformersBackend.from_pretrained(args.model, device=args.device)
    client = Decider(backend)
    rows = []
    for approx in [int(s) for s in args.states.split(",")]:
        state = state_of(approx)
        prefix = render_prefix(state)
        prefix_tokens = len(backend.tokenizer.encode(prefix))
        for k in (1, 20):
            _, branches = client._plan(questions(k))
            shared = timed(lambda: backend.score(prefix, branches), backend.device, args.repeats)
            naive = timed(lambda: [backend.score_uncached(prefix, b) for b in branches],
                          backend.device, args.repeats)
            backend.stats.prompt_tokens = backend.stats.cached_tokens = 0
            backend.score(prefix, branches)
            rows.append((prefix_tokens, k, shared / k * 1e3, naive / k * 1e3,
                         naive / shared, backend.stats.hit_rate))
            print(rows[-1], flush=True)

    lines = [
        f"# Prefix-sharing benchmark: {args.model} on {backend.device}",
        "",
        f"Median of {args.repeats} runs after one warm-up. `ms/q` is wall time per question.",
        "",
        "| state tokens | k | shared ms/q | naive ms/q | speedup | prefix hit rate |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    lines += [f"| {p} | {k} | {s:.1f} | {n:.1f} | {x:.2f}x | {h:.1%} |"
              for p, k, s, n, x, h in rows]
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
