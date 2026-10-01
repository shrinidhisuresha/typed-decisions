"""Labels vs zero-shot: how much does a little labelled data buy, and how fast is it?

    python examples/benchmark_labels.py --task banking77 --per-class 10
    python examples/benchmark_labels.py --task bitext_intent --approaches baseline-4B,probe-minilm-k50

Every approach is scored on the same split (typed_decisions.benchmarks: balanced sample,
halved per class). Each gets a temperature fitted on the train half and is reported on the
test half, with a paired 95% bootstrap against the zero-shot 4B baseline
(typed_decisions.gate.compare). Two tracks:

  zero-shot  only the text and the option names
  labels     also k labelled examples per class, from a pool that excludes every
             train/test row of the split

Raw distributions are cached under data/benchmark_dists/ (gitignored), so re-evaluating
after a metric or calibration change costs nothing.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import pathlib
import random
import time

import torch

from typed_decisions import Calibration, Decider, Prediction, diagnose_temperature, report
from typed_decisions.benchmarks import TASKS, download, load, read_rows, split
from typed_decisions.calibration.harness import _apply
from typed_decisions.gate import GatePolicy, compare
from typed_decisions.prompt import state_body

# Retrieval-cut options come back as exact zeros; fit and apply the temperature with the same
# floor (see typed_decisions.calibration.temperature.DEFAULT_FLOOR).
FLOOR = 0.005
# The zero-shot reference every approach is paired against: Qwen3.5-9B 4-bit on MLX (the
# `apple` preset). It runs in ~5 GB, so it measures reliably on a 24 GB laptop; the bfloat16 4B
# needs ~8 GB and thrashed under memory pressure. Override with --reference.
BASELINE = "mlx-9B-4bit"


def softmax(xs, scale=1.0):
    top = max(xs)
    e = [math.exp((x - top) * scale) for x in xs]
    return [v / sum(e) for v in e]


class Context:
    def __init__(self, task_name: str, pool_per_class: int, per_class: int, seed: int = 0):
        self.name = task_name
        self.task = TASKS[task_name]
        self.options = self.task.labels
        self.train, self.test = split(load(task_name, per_class))
        self.seed = seed
        self.pool_per_class = pool_per_class
        self._pool = None

    @property
    def pool(self) -> dict[str, list]:
        """Labelled rows NOT in the split, per class, seeded."""
        if self._pool is None:
            used = {json.dumps(r.state, sort_keys=True) for r in self.train + self.test}
            rows = read_rows(download(self.task.dataset))
            random.Random(self.seed).shuffle(rows)
            pool = {o: [] for o in self.options}
            for row in rows:
                label = self.task.label(row)
                if label in pool and len(pool[label]) < self.pool_per_class:
                    state = self.task.state(row)
                    if json.dumps(state, sort_keys=True) not in used:
                        pool[label].append(state)
            self._pool = pool
        return self._pool

    @property
    def instruction(self) -> str:
        return f"Find the answer option that best answers: {self.task.question.instructions}"


# --- embedders -----------------------------------------------------------------------------

class Embedder:
    """Sentence embeddings with the pooling and query prefix each model card specifies."""

    def __init__(self, name, pooling="last", prefix="", device=None):
        from transformers import AutoModel, AutoTokenizer
        from typed_decisions.backend_impls.transformers import best_device
        self.pooling, self.prefix = pooling, prefix
        self.device = best_device(device)
        side = "left" if pooling == "last" else "right"
        self.tok = AutoTokenizer.from_pretrained(name, padding_side=side)
        self.model = AutoModel.from_pretrained(name, dtype=torch.float32).to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, texts, batch=64):
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok([self.prefix + t for t in texts[i:i + batch]], padding=True,
                           truncation=True, max_length=256, return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state
            if self.pooling == "cls":
                v = h[:, 0]
            elif self.pooling == "last":
                v = h[:, -1]
            else:
                m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * m).sum(1) / m.sum(1)
            out.append(torch.nn.functional.normalize(v.float(), dim=1).cpu())
        return torch.cat(out)


EMBEDDERS = {
    "qwen0.6B": ("Qwen/Qwen3-Embedding-0.6B", "last", ""),
    "qwen4B": ("Qwen/Qwen3-Embedding-4B", "last", ""),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "mean", ""),
}


def train_probe(X, y, n_classes):
    """Multinomial logistic regression, L2-regularised, fitted with L-BFGS."""
    X = X.clone()   # embeddings come from inference_mode; autograd needs a normal tensor
    W = torch.zeros(X.shape[1], n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(X @ W * 20 + b, y) + 1e-3 * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach(), b.detach()


# --- approaches: ctx -> (train distributions, test distributions) ------------------------

def zero_shot_llm(model, permutations=4):
    """The decoder scorer: contextual calibration, retrieval above 26 options."""
    def run(ctx):
        from typed_decisions.backend_impls.transformers import TransformersBackend
        from typed_decisions.retrieval import EmbeddingRetriever
        if model.startswith("mlx-community/"):
            from typed_decisions.backend_impls.mlx import MLXBackend
            backend = MLXBackend.from_pretrained(model)
        else:
            backend = TransformersBackend.from_pretrained(model, dtype=torch.bfloat16)
        d = Decider(backend, calibration=Calibration(permutations=permutations, contextual=True),
                    retriever=EmbeddingRetriever(), shortlist_k=10)
        q = {"q": ctx.task.question}
        f = lambda rows: [[d.ask(x.state, q).answers["q"].probabilities[o] for o in ctx.options]
                          for x in rows]
        return f(ctx.train), f(ctx.test)
    return run


def similarity(embedder):
    """Zero-shot: cosine similarity between the text and each option name."""
    def run(ctx):
        from typed_decisions.retrieval import EmbeddingRetriever
        r = EmbeddingRetriever(EMBEDDERS[embedder][0])
        f = lambda rows: [softmax(r.rank(state_body(x.state), ctx.options, ctx.instruction), 50)
                          for x in rows]
        return f(ctx.train), f(ctx.test)
    return run


class Reranker:
    """Qwen3-Reranker, per its model card: p(yes) for (instruct, query, document)."""
    PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on "
              "the Query and the Instruct provided. Note that the answer can only be \"yes\" or "
              "\"no\".<|im_end|>\n<|im_start|>user\n")
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, model):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from typed_decisions.backend_impls.transformers import best_device
        self.device = best_device(None)
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.float16 if self.device != "cpu" else torch.float32
        ).to(self.device).eval()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")

    @torch.inference_mode()
    def log_odds(self, instruction, query, docs, batch=16):
        texts = [f"{self.PREFIX}<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {d}"
                 f"{self.SUFFIX}" for d in docs]
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok(texts[i:i + batch], padding=True, truncation=True, max_length=2048,
                           return_tensors="pt").to(self.device)
            last = self.model(**enc).logits[:, -1, :].float()
            out.extend((last[:, self.yes] - last[:, self.no]).tolist())
        return out


def reranker(model):
    def run(ctx):
        rr = Reranker(model)
        instruction = (f"{ctx.task.question.instructions} Does the Document name the correct "
                       "answer for this Query?")
        f = lambda rows: [softmax(rr.log_odds(instruction, state_body(x.state), ctx.options))
                          for x in rows]
        return f(ctx.train), f(ctx.test)
    return run


def centroids(embedder, k):
    """Labels: mean embedding of k examples per class; nearest centre."""
    def run(ctx):
        emb = Embedder(*EMBEDDERS[embedder])
        centres = []
        for o in ctx.options:
            c = emb([state_body(s) for s in ctx.pool[o][:k]]).mean(0)
            centres.append(c / c.norm())
        C = torch.stack(centres)
        f = lambda rows: [softmax((emb([state_body(x.state)])[0] @ C.T).tolist(), 50)
                          for x in rows]
        return f(ctx.train), f(ctx.test)
    return run


def probe(embedder, k):
    """Labels: a linear probe on frozen embeddings of k examples per class."""
    def run(ctx):
        emb = Embedder(*EMBEDDERS[embedder])
        pool = [(state_body(s), i) for i, o in enumerate(ctx.options) for s in ctx.pool[o][:k]]
        W, b = train_probe(emb([t for t, _ in pool]), torch.tensor([i for _, i in pool]),
                           len(ctx.options))
        f = lambda rows: torch.softmax(emb([state_body(x.state) for x in rows]) @ W * 20 + b,
                                       dim=1).tolist()
        return f(ctx.train), f(ctx.test)
    return run


def head(base, n):
    """Labels: a small encoder classification head trained on n examples per class."""
    def run(ctx):
        from typed_decisions.distill import TrainConfig, train_head
        from typed_decisions.traffic import Example, family_key
        q = ctx.task.question
        fam = family_key(q)
        uniform = [1.0 / len(ctx.options)] * len(ctx.options)
        examples = [Example(f"pool-{o}-{i}", fam, q, s, uniform, ctx.options.index(o))
                    for o in ctx.options for i, s in enumerate(ctx.pool[o][:n])]
        h = train_head(examples, base=base, config=TrainConfig(epochs=3, lr=5e-5, batch_size=32,
                                                               max_length=256))
        f = lambda rows: h.predict_proba([x.state for x in rows], temperature=1.0)
        return f(ctx.train), f(ctx.test)
    return run


APPROACHES = {
    "baseline-4B": ("zero-shot", zero_shot_llm("Qwen/Qwen3.5-4B")),
    "mlx-9B-4bit": ("zero-shot", zero_shot_llm("mlx-community/Qwen3.5-9B-4bit")),
    "emb-0.6B": ("zero-shot", similarity("qwen0.6B")),
    "emb-4B": ("zero-shot", similarity("qwen4B")),
    "rerank-0.6B": ("zero-shot", reranker("Qwen/Qwen3-Reranker-0.6B")),
    "rerank-4B": ("zero-shot", reranker("Qwen/Qwen3-Reranker-4B")),
    "centroid-emb0.6B-k10": ("labels", centroids("qwen0.6B", 10)),
    "probe-emb0.6B-k10": ("labels", probe("qwen0.6B", 10)),
    "probe-emb0.6B-k50": ("labels", probe("qwen0.6B", 50)),
    "probe-minilm-k10": ("labels", probe("minilm", 10)),
    "probe-minilm-k50": ("labels", probe("minilm", 50)),
    "head-base-50": ("labels", head("answerdotai/ModernBERT-base", 50)),
    "head-base-200": ("labels", head("answerdotai/ModernBERT-base", 200)),
}


def evaluate(ctx, train_d, test_d):
    fit = diagnose_temperature([Prediction(dict(zip(ctx.options, d)), r.label)
                                for d, r in zip(train_d, ctx.train)], floor=FLOOR)
    t = fit.temperature if fit.trustworthy else 1.0
    preds = _apply([Prediction(dict(zip(ctx.options, d)), r.label)
                    for d, r in zip(test_d, ctx.test)], t, FLOOR)
    return t, report(preds), [[p.probabilities[o] for o in ctx.options] for p in preds]


def distributions(ctx, name, args):
    """Cached raw train/test distributions for one approach."""
    path = pathlib.Path("data/benchmark_dists") / (
        f"{ctx.name}-pc{args.per_class}-pool{args.pool}-{name}.json")
    if path.exists():
        return json.loads(path.read_text())
    train_d, test_d = APPROACHES[name][1](ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([train_d, test_d]))
    return train_d, test_d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="banking77")
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--pool", type=int, default=200, help="labelled pool rows per class")
    parser.add_argument("--approaches", default=",".join(APPROACHES))
    parser.add_argument("--reference", default=BASELINE)
    parser.add_argument("--out")
    args = parser.parse_args()
    reference = args.reference
    ctx = Context(args.task, args.pool, args.per_class)
    out = args.out or f"docs/benchmark_{args.task}.md"
    outcomes = [ctx.options.index(r.label) for r in ctx.test]

    _, base_rep, base = evaluate(ctx, *distributions(ctx, reference, args))
    lines = [
        f"# Labels vs zero-shot on {args.task}: {len(ctx.options)} options, "
        f"{len(ctx.test)} test rows", "",
        "Temperature fitted on the train half, metrics on the test half, paired 95% bootstrap "
        f"against the zero-shot reference ({reference}; zero-shot rows use 4 orderings, "
        "contextual calibration, retrieval above 26 options). `labels` approaches draw from a pool that "
        "excludes every row of the split.", "",
        "| approach | track | T | acc | ECE | Brier | acc diff vs reference | Brier diff vs reference "
        "| right only here / only reference | s |",
        "|---|---|---:|---:|---:|---:|---|---|---|---:|",
        f"| **{reference}** (reference) | zero-shot | | {base_rep.accuracy:.3f} | {base_rep.ece:.3f} | "
        f"{base_rep.brier:.3f} | | | | |",
    ]
    print(lines[-1], flush=True)
    for name in [a for a in args.approaches.split(",") if a != reference]:
        track = APPROACHES[name][0]
        start = time.perf_counter()
        try:
            train_d, test_d = distributions(ctx, name, args)
        except Exception as exc:  # one approach failing (OOM, download) must not lose the rest
            lines.append(f"| {name} | {track} | | | | | failed: {type(exc).__name__} | | | |")
            continue
        t, rep, dists = evaluate(ctx, train_d, test_d)
        v = compare(dists, base, outcomes, GatePolicy(min_eval=0))
        a, b = v["accuracy_diff_ci"], v["brier_diff_ci"]
        lines.append(
            f"| {name} | {track} | {t:.2f} | {rep.accuracy:.3f} | {rep.ece:.3f} | {rep.brier:.3f} "
            f"| [{a[0]:+.3f}, {a[1]:+.3f}] | [{b[0]:+.3f}, {b[1]:+.3f}] "
            f"| {v['student_only_right']} / {v['teacher_only_right']} "
            f"| {time.perf_counter() - start:.0f} |")
        print(lines[-1], flush=True)
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
