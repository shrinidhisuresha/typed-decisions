"""Tuned adapter vs the inference-time pipeline, on datasets the adapter never saw.

    python -m typed_decisions.finetune.eval --adapter adapters/b77 --out docs/finetune_qwen3.5-0.8b.md

Configs, all scored through the real Decider so the prompts are the served ones:
  raw      base model, 1 pass, T=1
  posthoc  base model, permutations=4 + contextual, T fitted on the train half
  tuned    base + adapter, 1 pass, T=1 (the claim: no fitting, no extra passes)
The fitted-T column is reported for every config; for `tuned` it is diagnostic only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..backend_impls.transformers import TransformersBackend, best_device
from ..benchmarks import TASKS, load, split
from ..calibration.harness import Prediction, _apply, fit_temperature_from_predictions, report
from ..client import Calibration, Decider
from ..gate import GatePolicy, compare
from ..types import NoulQuestion

PER_CLASS = {"bitext_category": 40, "tickets_incident": 150, "tickets_priority": 50,
             "bitext_telco_category": 40}
CONFIGS = {
    "raw": Calibration(),
    "posthoc": Calibration(permutations=4, contextual=True),
    "tuned": Calibration(),
}


def distribution(task, answer) -> dict[str, float]:
    if isinstance(task.question, NoulQuestion):
        return {"yes": answer.noul, "no": 1.0 - answer.noul}
    return dict(answer.probabilities)


def predictions(client: Decider, task, rows) -> list[Prediction]:
    return [Prediction(distribution(task, client.ask(r.state, {"q": task.question}).answers["q"]),
                       r.label) for r in rows]


def backend(base: str, adapter: str | None, device: str) -> TransformersBackend:
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.to(device).eval()
    return TransformersBackend(model, AutoTokenizer.from_pretrained(base), device)


def as_lists(task, preds: list[Prediction]) -> tuple[list[list[float]], list[int]]:
    return ([[p.probabilities[o] for o in task.labels] for p in preds],
            [task.labels.index(p.label) for p in preds])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--adapter")
    parser.add_argument("--tasks", default=",".join(PER_CLASS))
    parser.add_argument("--configs", default="raw,posthoc,tuned")
    parser.add_argument("--device")
    parser.add_argument("--dists", default="data/finetune_dists")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    device = best_device(args.device)
    configs = args.configs.split(",")
    if "tuned" in configs and not args.adapter:
        parser.error("config 'tuned' needs --adapter")
    dists = Path(args.dists)
    dists.mkdir(parents=True, exist_ok=True)
    tag = Path(args.adapter).name if args.adapter else "base"

    results: dict[str, dict[str, dict]] = {}
    for config in configs:
        b = backend(args.base, args.adapter if config == "tuned" else None, device)
        client = Decider(b, calibration=CONFIGS[config])
        for name in args.tasks.split(","):
            task = TASKS[name]
            train, test = split(load(name, PER_CLASS.get(name, 20)))
            path = dists / f"{name}-{config}-{tag if config == 'tuned' else 'base'}.json"
            if path.exists():
                cached = json.loads(path.read_text())
                fit_on = [Prediction(**p) for p in cached["train"]]
                held = [Prediction(**p) for p in cached["test"]]
            else:
                fit_on, held = predictions(client, task, train), predictions(client, task, test)
                path.write_text(json.dumps({k: [{"probabilities": dict(p.probabilities),
                                                 "label": p.label} for p in v]
                                            for k, v in (("train", fit_on), ("test", held))}))
            t = fit_temperature_from_predictions(fit_on)
            results.setdefault(name, {})[config] = {
                "raw": report(held), "fitted": report(held, temperature=t), "T": t,
                "test": held,
            }
            print(f"{name:18s} {config:8s} {report(held).summary()}  | T={t:.3f} "
                  f"{report(held, temperature=t).summary()}", flush=True)
        del client, b
        if device == "mps":
            torch.mps.empty_cache()

    lines = [f"# Outcome-calibrated fine-tuning: {args.base}", "",
             f"Adapter: `{args.adapter}`. Every task below is **held out**: the adapter never "
             "trained on its dataset. Each task is split in half per class; T is fitted on "
             "the train half and every number is on the test half.", ""]
    for name, by_config in results.items():
        task = TASKS[name]
        n = next(iter(by_config.values()))["raw"].count
        lines += [f"## {name} ({task.question.type}, {len(task.labels)} options, n={n} test)", "",
                  "| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for config, r in by_config.items():
            passes = "4 + null" if config == "posthoc" else "1"
            lines.append(f"| {config} | {passes} | {r['raw'].accuracy:.3f} | {r['raw'].ece:.3f} | "
                         f"{r['raw'].brier:.3f} | {r['T']:.3f} | {r['fitted'].ece:.3f} | "
                         f"{r['fitted'].brier:.3f} |")
        if "tuned" in by_config and "posthoc" in by_config:
            tuned = by_config["tuned"]["test"]
            post = _apply(by_config["posthoc"]["test"], by_config["posthoc"]["T"])
            s, outcomes = as_lists(task, tuned)
            p, _ = as_lists(task, post)
            v = compare(s, p, outcomes, GatePolicy())
            lines += ["", f"**tuned (1 pass, T=1) vs posthoc (fitted T)**, paired 95% bootstrap: "
                      f"Brier diff [{v['brier_diff_ci'][0]:+.3f}, {v['brier_diff_ci'][1]:+.3f}], "
                      f"accuracy diff [{v['accuracy_diff_ci'][0]:+.3f}, "
                      f"{v['accuracy_diff_ci'][1]:+.3f}] -> gate: "
                      f"{'PASS' if v['promoted'] else 'FAIL'} ({'; '.join(v['reasons'])})"]
        lines.append("")
    Path(args.out).write_text("\n".join(lines))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
