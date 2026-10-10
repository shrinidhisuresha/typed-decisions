"""The same objective as train.py, on MLX: QLoRA over 4-bit weights for 4B+ models on a Mac.

    python -m typed_decisions.finetune.train_mlx --base mlx-community/Qwen3.5-4B-4bit \\
        --mixture mix2 --out adapters/mix2-4b-r16

mlx_lm's own LoRA trainer optimises the full-vocabulary next-token loss, which is the
wrong objective here, so this is a custom loop: label-restricted log score (+ RPS on Score
rows), exactly as in train.py. The adapter saves in mlx_lm's format, so
`mlx_lm.load(base, adapter_path=out)` and eval.py --backend mlx pick it up.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from ..backend_impls.mlx import _Tokenizer
from ..benchmarks import TASKS, load
from .data import Encoded, build, build_mixture, encode
from .sources import MIXTURES

LORA_KEYS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "linear_attn.in_proj_qkv", "linear_attn.out_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def collate(batch: list[Encoded], pad_id: int):
    width = max(len(e.input_ids) for e in batch)
    k = max(len(e.label_ids) for e in batch)
    ids, last, labels, mask, target = [], [], [], [], []
    for e in batch:
        n, m = len(e.input_ids), len(e.label_ids)
        ids.append(e.input_ids + [pad_id] * (width - n))   # right padding: causal, so the
        last.append(n - 1)                                  # last real token never sees it
        labels.append(e.label_ids + [0] * (k - m))
        mask.append([True] * m + [False] * (k - m))
        target.append(e.target + [0.0] * (k - m))
    return (mx.array(ids), mx.array(last), mx.array(labels), mx.array(mask),
            mx.array(target, dtype=mx.float32))


def answer_logits(model, ids, last):
    """Logits at each row's `Answer:` position only. Running the full model would build a
    (batch, seq, 248k) logit tensor plus its gradient -- the 4B smoke run swapped 23 GB."""
    text = getattr(model, "language_model", model)
    hidden = text.model(ids)[mx.arange(ids.shape[0]), last]
    if text.args.tie_word_embeddings:
        return text.model.embed_tokens.as_linear(hidden)
    return text.lm_head(hidden)


def loss_fn(model, ids, last, labels, mask, target, ordinal, rps_weight, penalised=None,
            penalty=0.0):
    picked = answer_logits(model, ids, last).astype(mx.float32)
    picked = mx.take_along_axis(picked, labels, axis=1)
    picked = mx.where(mask, picked, -mx.inf)
    logp = picked - mx.logsumexp(picked, axis=1, keepdims=True)
    loss = -(target * mx.where(mask, logp, 0.0)).sum(axis=1)
    if rps_weight:
        rps = []
        for i, idx in enumerate(ordinal):
            if not idx:
                rps.append(mx.array(0.0))
                continue
            ix = mx.array(list(idx))
            p, t = mx.exp(logp[i][ix]), target[i][ix]
            rps.append(((mx.cumsum(p) - mx.cumsum(t)) ** 2).sum() / max(len(idx) - 1, 1))
        loss = loss + rps_weight * mx.stack(rps)
    if penalty and penalised is not None:
        # confidence penalty on Noul/Score rows only; see loss.entropy
        safe = mx.where(mask, logp, 0.0)
        h = -(mx.exp(safe) * safe * mask).sum(axis=1)
        loss = loss - penalty * penalised * h
    return loss.mean()


def main(argv=None) -> None:
    from mlx_lm import load as mlx_load
    from mlx_lm.tuner.trainer import grad_checkpoint
    from mlx_lm.tuner.utils import linear_to_lora_layers

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="mlx-community/Qwen3.5-4B-4bit")
    parser.add_argument("--task", default="banking77")
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--mixture", choices=sorted(MIXTURES))
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--rps-weight", type=float, default=0.0)
    parser.add_argument("--confidence-penalty", type=float, default=0.0,
                        help="subtract this times H(p) on Noul and Score rows")
    parser.add_argument("--penalty-start", type=int, default=0,
                        help="step at which the confidence penalty switches on; -1 means "
                             "the end of warmup. With the penalty on from step 0, 2 of 3 4B seeds "
                             "collapsed to uniform answers at the LR peak")
    parser.add_argument("--abort-loss", type=float, default=0.0,
                        help="after warmup, stop with exit code 3 if the 20-step mean loss "
                             "exceeds this (the collapse signature); 0 disables")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--null-share", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memory-limit-gb", type=float, default=10.0,
                        help="MLX allocation cap (soft: MLX may still exceed it)")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="drop longer rows: in training mlx_lm runs Qwen3.5's linear "
                             "attention as a per-token loop whose saved state grows with length")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    mx.set_memory_limit(int(args.memory_limit_gb * 2**30))
    mx.set_cache_limit(2 * 2**30)
    mx.random.seed(args.seed)
    model, tok = mlx_load(args.base)
    tokenizer = _Tokenizer(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    if args.mixture:
        rows = build_mixture(MIXTURES[args.mixture], seed=args.seed, null_share=args.null_share,
                             scale=args.scale)
        source = f"mixture {args.mixture}"
    else:
        rows = build(load(args.task, args.per_class), TASKS[args.task].labels, seed=args.seed,
                     null_share=args.null_share)
        source = f"{args.task} ({args.per_class}/class)"
    encoded = [encode(r, tokenizer) for r in rows]
    kept = [e for e in encoded if len(e.input_ids) <= args.max_tokens]
    print(f"{len(rows)} rows from {source}; {len(kept)} within {args.max_tokens} tokens",
          flush=True)
    encoded = kept

    model.freeze()
    lora = {"rank": args.rank, "scale": 2.0, "dropout": 0.05, "keys": LORA_KEYS}
    linear_to_lora_layers(model, len(model.layers), lora)
    grad_checkpoint(model.layers[0])          # recompute each block's activations in backward
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"trainable params: {n_train:,}", flush=True)
    model.train()

    # The schedule always spans the full run; --max-steps only stops early, so a short
    # stability test sees exactly the LR curve (warmup, peak) the full run would.
    steps = math.ceil(len(encoded) / args.batch) * args.epochs
    stop_at = min(steps, args.max_steps) if args.max_steps else steps
    warmup = max(1, steps // 20)
    penalty_start = warmup if args.penalty_start < 0 else args.penalty_start
    schedule = optim.join_schedules(
        [optim.linear_schedule(args.lr / warmup, args.lr, warmup),
         optim.linear_schedule(args.lr, 0.0, steps - warmup)], [warmup])
    opt = optim.AdamW(learning_rate=schedule, weight_decay=0.0)
    value_and_grad = nn.value_and_grad(model, loss_fn)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history, step, start = [], 0, time.perf_counter()
    rng = random.Random(args.seed)
    for _ in range(args.epochs):
        order = list(range(len(encoded)))
        rng.shuffle(order)
        for i in range(0, len(order), args.batch):
            if step >= stop_at:
                break
            batch = [encoded[j] for j in order[i:i + args.batch]]
            total, grads = 0.0, None
            for m in range(0, len(batch), args.micro_batch):
                part = batch[m:m + args.micro_batch]
                loss, g = value_and_grad(model, *collate(part, pad_id),
                                         [e.ordinal for e in part], args.rps_weight,
                                         mx.array([float(e.penalised) for e in part]),
                                         args.confidence_penalty if step >= penalty_start
                                         else 0.0)
                w = len(part) / len(batch)
                g = tree_map(lambda x: x * w, g)
                grads = g if grads is None else tree_map(mx.add, grads, g)
                total += loss.item() * w
            grads, _ = optim.clip_grad_norm(grads, 1.0)
            opt.update(model, grads)
            mx.eval(model.trainable_parameters(), opt.state)
            history.append(total)
            step += 1
            recent = sum(history[-20:]) / len(history[-20:])
            if args.abort_loss and step > warmup + 20 and recent > args.abort_loss:
                print(f"ABORT step {step}: 20-step mean loss {recent:.4f} > {args.abort_loss} "
                      "after warmup (collapse)", flush=True)
                raise SystemExit(3)
            if step % 20 == 0 or step == stop_at:
                print(f"step {step}/{steps}  loss {recent:.4f}  "
                      f"{(time.perf_counter() - start) / step:.2f} s/step  "
                      f"peak {mx.get_peak_memory() / 2**30:.1f} GB", flush=True)

    mx.save_safetensors(str(out / "adapters.safetensors"),
                        dict(tree_flatten(model.trainable_parameters())))
    (out / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora", "num_layers": len(model.layers), "lora_parameters": lora}))
    (out / "train.json").write_text(json.dumps({
        "base": args.base, "mixture": args.mixture, "scale": args.scale,
        "rps_weight": args.rps_weight, "confidence_penalty": args.confidence_penalty,
        "penalty_start": penalty_start, "warmup": warmup, "rows": len(rows), "kept": len(encoded),
        "max_tokens": args.max_tokens, "epochs": args.epochs,
        "batch": args.batch, "micro_batch": args.micro_batch, "lr": args.lr, "rank": args.rank,
        "seed": args.seed, "steps": step, "seconds": round(time.perf_counter() - start, 1),
        "loss": history}, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
