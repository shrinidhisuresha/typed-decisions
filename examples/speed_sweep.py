"""Speed sweep: how small can the labelled path's embedding model get before accuracy suffers?

    python examples/speed_sweep.py [--out docs/speed_sweep.md]

On BANKING77's official splits (the same setup as examples/hf_banking77_probe.py), for each
embedding model: train the linear probe on k examples per class sampled from the official
train split, report accuracy, ECE and Brier on the official 3,080-row test split, then time one
query end to end (embed + linear + softmax), p50/p99, on the Apple GPU (mps) and on the CPU.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from benchmark_labels import Embedder, train_probe  # noqa: E402
from hf_banking77_probe import official_splits  # noqa: E402

from typed_decisions import Prediction, report  # noqa: E402

# (model, pooling, query prefix). Pooling and prefixes follow each model card.
MODELS = [
    ("sentence-transformers/all-MiniLM-L6-v2", "mean", ""),
    ("ibm-granite/granite-embedding-30m-english", "cls", ""),
    ("BAAI/bge-small-en-v1.5", "cls", ""),
    ("intfloat/e5-small-v2", "mean", "query: "),
    ("BAAI/bge-base-en-v1.5", "cls", ""),
    ("Qwen/Qwen3-Embedding-0.6B", "last", ""),
]


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def latency(embedder, W, b, texts, device, runs=200):
    fn = lambda t: torch.softmax(embedder([t]) @ W * 20 + b, dim=1)
    for t in texts[:5]:
        fn(t)
    samples = []
    for i in range(runs):
        sync(device)
        start = time.perf_counter_ns()
        fn(texts[i % len(texts)])
        sync(device)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    s = sorted(samples)
    return statistics.median(s), s[int(0.99 * (len(s) - 1))]


def main() -> None:
    import random
    from collections import defaultdict
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="speed_sweep.md")
    args = parser.parse_args()
    train, test = official_splits()
    labels = sorted({r["category"] for r in train})
    index = {c: i for i, c in enumerate(labels)}
    by_class = defaultdict(list)
    for r in train:
        by_class[r["category"]].append(r["text"])
    rng = random.Random(0)
    for c in by_class:
        rng.shuffle(by_class[c])
    texts = [r["text"] for r in test]
    ytest = [r["category"] for r in test]
    lines = [
        "# Speed sweep: embedding size vs accuracy and latency (BANKING77, 77 intents)", "",
        "Linear probe on k examples per class from the official train split; metrics on the "
        "official 3,080-row test split. Latency is one query end to end (embed + linear + "
        "softmax), Apple M5 Pro, 200 runs after warm-up.", "",
        "| embedding model | params | k | acc | ECE | Brier | mps p50 / p99 ms | cpu p50 / p99 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, pooling, prefix in MODELS:
        timings, results = {}, []
        for device in ("mps", "cpu"):
            emb = Embedder(name, pooling, prefix, device)
            if device == "mps":
                params = sum(p.numel() for p in emb.model.parameters())
                Xte = emb(texts)
                for k in (10, 50):
                    pool = [(t, index[c]) for c in labels for t in by_class[c][:k]]
                    W, b = train_probe(emb([t for t, _ in pool]),
                                       torch.tensor([i for _, i in pool]), len(labels))
                    probs = torch.softmax(Xte @ W * 20 + b, dim=1).tolist()
                    rep = report([Prediction(dict(zip(labels, p)), y)
                                  for p, y in zip(probs, ytest)])
                    results.append((k, rep, W, b))
            W, b = results[-1][2], results[-1][3]
            timings[device] = latency(emb, W, b, texts, device)
            del emb
        for k, rep, _, _ in results:
            lines.append(
                f"| {name.split('/')[-1]} | {params / 1e6:.0f}M | {k} | {rep.accuracy:.3f} | "
                f"{rep.ece:.3f} | {rep.brier:.3f} | "
                f"{timings['mps'][0]:.1f} / {timings['mps'][1]:.1f} | "
                f"{timings['cpu'][0]:.1f} / {timings['cpu'][1]:.1f} |")
            print(lines[-1], flush=True)
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
