"""Train, evaluate and package the BANKING77 intent probe for the Hugging Face Hub.

    python examples/hf_banking77_probe.py --out hf/banking77-intent-probe-minilm

A linear probe on frozen all-MiniLM-L6-v2 embeddings, trained on BANKING77's official train
split (all of it, and 10 / 50 examples per class for the few-shot rows) and evaluated on the
official 3,080-row test split. Writes the chosen probe as safetensors + config + a model card
that names the base model, the dataset, their licences, the metrics and the citation.

Nothing is uploaded; publishing is a separate, manual step.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(__file__))
from benchmark_labels import Embedder, train_probe  # noqa: E402

from typed_decisions import Prediction, diagnose_temperature, report  # noqa: E402
from typed_decisions.benchmarks import download, read_rows  # noqa: E402

BASE = ("sentence-transformers/all-MiniLM-L6-v2", "mean", "")
SCALE = 20.0   # logit scale used in training (cosine-normalised embeddings)


def official_splits():
    train_path, test_path = download("PolyAI/banking77")
    return read_rows(train_path), read_rows(test_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="hf/banking77-intent-probe-minilm")
    parser.add_argument("--publish-k", default="full", help="which probe to package: 10, 50 or full")
    args = parser.parse_args()

    train, test = official_splits()
    labels = sorted({r["category"] for r in train})
    index = {c: i for i, c in enumerate(labels)}
    emb = Embedder(*BASE, device="cpu")
    Xtest = emb([r["text"] for r in test])
    ytest = [r["category"] for r in test]

    by_class = defaultdict(list)
    for r in train:
        by_class[r["category"]].append(r["text"])
    rng = random.Random(0)
    for c in by_class:
        rng.shuffle(by_class[c])

    results, probes = [], {}
    for k in ("10", "50", "full"):
        texts = [(t, index[c]) for c in labels
                 for t in (by_class[c] if k == "full" else by_class[c][:int(k)])]
        X = emb([t for t, _ in texts])
        W, b = train_probe(X, torch.tensor([i for _, i in texts]), len(labels))
        probs = torch.softmax(Xtest @ W * SCALE + b, dim=1).tolist()
        preds = [Prediction(dict(zip(labels, p)), y) for p, y in zip(probs, ytest)]
        rep = report(preds)
        results.append((k, len(texts), rep))
        probes[k] = (W, b)
        print(f"k={k:>4}  train={len(texts):>5}  acc={rep.accuracy:.4f}  "
              f"ECE={rep.ece:.4f}  Brier={rep.brier:.4f}", flush=True)

    W, b = probes[args.publish_k]
    # latency on this machine's CPU: one query end to end
    one = lambda t: torch.softmax(emb([t]) @ W * SCALE + b, dim=1)
    for t in [r["text"] for r in test[:5]]:
        one(t)
    ms = []
    for r in test[:300]:
        start = time.perf_counter_ns()
        one(r["text"])
        ms.append((time.perf_counter_ns() - start) / 1e6)
    p50, p99 = statistics.median(ms), sorted(ms)[int(0.99 * (len(ms) - 1))]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    save_file({"weight": W.contiguous(), "bias": b.contiguous()}, str(out / "probe.safetensors"))
    (out / "config.json").write_text(json.dumps({
        "base_model": BASE[0], "pooling": BASE[1], "query_prefix": BASE[2], "normalize": True,
        "logit_scale": SCALE, "labels": labels, "trained_on": f"BANKING77 train, k={args.publish_k}",
    }, indent=2))

    rows = "\n".join(f"| {'all' if k == 'full' else k} | {n:,} | {rep.accuracy:.3f} | "
                     f"{rep.ece:.3f} | {rep.brier:.3f} |" for k, n, rep in results)
    card = f"""---
license: apache-2.0
base_model: sentence-transformers/all-MiniLM-L6-v2
datasets:
- PolyAI/banking77
language:
- en
pipeline_tag: text-classification
library_name: typed-decisions
tags:
- intent-classification
- linear-probe
- few-shot
- calibration
metrics:
- accuracy
---

# BANKING77 intent probe (all-MiniLM-L6-v2)

A **linear probe** (77-way logistic regression) on frozen
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) sentence
embeddings, for the 77 intents of [BANKING77](https://huggingface.co/datasets/PolyAI/banking77).
Built with [typed-decisions](https://github.com/shrinidhisuresha/typed-decisions). It returns a
calibrated probability for every intent, and it's small (about {os.path.getsize(out / 'probe.safetensors') // 1024} KB on top of the 23M base) and fast.

## Evaluation (official BANKING77 test split, 3,080 queries)

| labelled examples per class | train size | accuracy | ECE | Brier |
|---|---:|---:|---:|---:|
{rows}

This repository contains the **{'all-data' if args.publish_k == 'full' else args.publish_k + '-per-class'}** probe.
Latency, one query end to end (embed + probe) on an Apple M5 Pro **CPU**: p50 **{p50:.1f} ms**,
p99 {p99:.1f} ms.

## Usage

```python
import json, torch
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

cfg = json.load(open("config.json")); probe = load_file("probe.safetensors")
tok = AutoTokenizer.from_pretrained(cfg["base_model"]); enc = AutoModel.from_pretrained(cfg["base_model"])
batch = tok(["I was charged twice for one purchase"], return_tensors="pt", padding=True)
h = enc(**batch).last_hidden_state; m = batch["attention_mask"].unsqueeze(-1)
v = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=1)
p = torch.softmax(v @ probe["weight"] * cfg["logit_scale"] + probe["bias"], dim=1)[0]
print(cfg["labels"][int(p.argmax())], float(p.max()))
```

## Training

- Frozen base, mean pooling, L2-normalised embeddings.
- Multinomial logistic regression with L2 penalty 1e-3 and a logit scale of {SCALE:.0f}, fitted
  with L-BFGS.
- Few-shot rows use a seeded random sample per class from the official train split.
- No test data was used for training or model selection.

## Limitations

- **English only.** Intents are the 77 fixed BANKING77 classes, a single domain (card and
  account banking).
- **Out-of-scope queries still get one of the 77 intents.** Gate on the probability if that
  matters.
- **Calibration** was measured on the BANKING77 test split. Re-check it on your own traffic.

## Licences and attribution

- Probe weights: Apache-2.0.
- Base model: all-MiniLM-L6-v2 (Apache-2.0).
- Training data: BANKING77, **CC BY 4.0**. Please cite:

> Casanueva, I., Temčinas, T., Gerz, D., Henderson, M., Vulić, I. (2020). *Efficient Intent
> Detection with Dual Sentence Encoders.* Proceedings of the 2nd Workshop on NLP for
> Conversational AI.
"""
    (out / "README.md").write_text(card)
    print(f"wrote {out}/ (probe.safetensors, config.json, README.md); CPU p50 {p50:.1f} ms")


if __name__ == "__main__":
    main()
