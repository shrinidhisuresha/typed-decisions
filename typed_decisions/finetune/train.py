"""Fit a LoRA adapter on the label-restricted log score.

    python -m typed_decisions.finetune.train --base Qwen/Qwen3.5-0.8B --per-class 30 --out adapters/b77

Trains on BANKING77 only by default. eval.py then measures it on datasets it never saw.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..backend_impls.transformers import _HFTokenizer, best_device
from ..benchmarks import TASKS, load
from .data import Encoded, build, build_mixture, encode
from .sources import MIXTURES
from .loss import entropy, log_score, ordinal_rps, restricted_log_probs

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "out_proj",
                "gate_proj", "up_proj", "down_proj"]


def collate(batch: list[Encoded], pad_id: int, device: str):
    """Right-padded ids, the index of each row's last real token, padded label ids."""
    width = max(len(e.input_ids) for e in batch)
    k = max(len(e.label_ids) for e in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), width), dtype=torch.long)
    labels = torch.zeros((len(batch), k), dtype=torch.long)
    mask = torch.zeros((len(batch), k), dtype=torch.bool)
    target = torch.zeros((len(batch), k))
    for i, e in enumerate(batch):
        n, m = len(e.input_ids), len(e.label_ids)
        ids[i, :n] = torch.tensor(e.input_ids)
        attn[i, :n] = 1
        labels[i, :m] = torch.tensor(e.label_ids)
        mask[i, :m] = True
        target[i, :m] = torch.tensor(e.target)
    last = attn.sum(1) - 1
    return [t.to(device) for t in (ids, attn, last, labels, mask, target)]


def batch_loss(model, batch, pad_id: int, device: str, rps_weight: float = 0.0,
               penalty: float = 0.0) -> torch.Tensor:
    ids, attn, last, labels, mask, target = collate(batch, pad_id, device)
    inner = model.get_base_model()
    hidden = inner.model(input_ids=ids, attention_mask=attn).last_hidden_state
    # The lm_head only at the `Answer:` position: full-sequence logits over a 248k vocab
    # would dominate memory for no benefit.
    logits = inner.lm_head(hidden[torch.arange(len(batch), device=device), last])
    logp = restricted_log_probs(logits, labels, mask)
    loss = log_score(logp, target, mask)
    if rps_weight:
        loss = loss + rps_weight * ordinal_rps(logp, target, [e.ordinal for e in batch])
    if penalty:
        which = torch.tensor([e.penalised for e in batch], device=device, dtype=loss.dtype)
        loss = loss - penalty * which * entropy(logp, mask)
    return loss.mean()


def main(argv=None) -> None:
    from peft import LoraConfig, get_peft_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--task", default="banking77")
    parser.add_argument("--mixture", choices=sorted(MIXTURES),
                        help="train on several sources and question shapes instead of --task")
    parser.add_argument("--scale", type=float, default=1.0, help="multiply every per-class count")
    parser.add_argument("--rps-weight", type=float, default=0.0,
                        help="add this times RPS on Score rows (ordinal-aware, also proper)")
    parser.add_argument("--confidence-penalty", type=float, default=0.0,
                        help="subtract this times H(p) on Noul and Score rows")
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, default=2,
                        help="rows per forward; --batch is reached by accumulation")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--noul-share", type=float, default=0.3)
    parser.add_argument("--null-share", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=0, help="stop early (smoke runs)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = best_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.base)
    tokenizer = _HFTokenizer(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    if args.mixture:
        rows = build_mixture(MIXTURES[args.mixture], seed=args.seed, null_share=args.null_share,
                             scale=args.scale)
        source = f"mixture {args.mixture}"
    else:
        task = TASKS[args.task]
        rows = build(load(args.task, args.per_class), task.labels, seed=args.seed,
                     noul_share=args.noul_share, null_share=args.null_share)
        source = f"{args.task} ({args.per_class}/class)"
    encoded = [encode(r, tokenizer) for r in rows]
    shapes = {}
    for r in rows:
        shapes[r.question.type] = shapes.get(r.question.type, 0) + 1
    print(f"{len(rows)} rows from {source} {shapes}, device {device}")

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16).to(device)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank,
                                             lora_dropout=0.05, target_modules=LORA_TARGETS))
    # Without checkpointing the reference (non-fused) linear-attention kernels keep every
    # chunk's activations for backward: 13 GB at batch 8, enough to swap a 24 GB Mac.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    model.train()

    steps = math.ceil(len(encoded) / args.batch) * args.epochs
    if args.max_steps:
        steps = min(steps, args.max_steps)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                              weight_decay=0.0)
    warmup = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, 1 - s / steps))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history, step, start = [], 0, time.perf_counter()
    rng = random.Random(args.seed)
    for _ in range(args.epochs):
        order = list(range(len(encoded)))
        rng.shuffle(order)
        for i in range(0, len(order), args.batch):
            if step >= steps:
                break
            rows = [encoded[j] for j in order[i:i + args.batch]]
            loss = 0.0
            for m in range(0, len(rows), args.micro_batch):
                part = rows[m:m + args.micro_batch]
                micro = (batch_loss(model, part, pad_id, device, args.rps_weight,
                                     args.confidence_penalty)
                         * len(part) / len(rows))
                micro.backward()
                loss += micro.detach()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            history.append(loss.item())
            step += 1
            if step % 20 == 0 and device == "mps":
                torch.mps.empty_cache()
            if step % 20 == 0 or step == steps:
                recent = sum(history[-20:]) / len(history[-20:])
                print(f"step {step}/{steps}  loss {recent:.4f}  "
                      f"{(time.perf_counter() - start) / step:.2f} s/step"
                      + (f"  mps {torch.mps.driver_allocated_memory() / 2**30:.1f} GB"
                         if device == "mps" else ""), flush=True)

    model.save_pretrained(out)
    (out / "train.json").write_text(json.dumps({
        "base": args.base, "task": args.task, "mixture": args.mixture, "scale": args.scale,
        "rps_weight": args.rps_weight, "confidence_penalty": args.confidence_penalty,
        "per_class": args.per_class, "rows": len(rows),
        "epochs": args.epochs, "batch": args.batch, "micro_batch": args.micro_batch, "lr": args.lr, "rank": args.rank,
        "noul_share": args.noul_share, "null_share": args.null_share, "seed": args.seed,
        "steps": step, "seconds": round(time.perf_counter() - start, 1), "loss": history,
    }, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
