"""Verify an HTTP backend against the reference TransformersBackend on a live server.

    vllm serve Qwen/Qwen2.5-0.5B-Instruct --host 127.0.0.1 --port 8811 --dtype float32 \\
        --enable-prefix-caching --enable-prompt-tokens-details --logprobs-mode processed_logprobs
    python examples/verify_backend.py vllm Qwen/Qwen2.5-0.5B-Instruct http://127.0.0.1:8811

Scores the same prompts -- letter branches and option-text (sequence) branches -- through
the server and through the local reference, then compares the per-option probabilities
each implies. The HTTP backends were written against mocks; this is the check that they
read the right numbers off a real server. Also prints the server-reported prefix hit rate.

For vLLM both modes are checked: the default (exact, prompt logprobs, no cache) and
cache_letters=True. Enable cache_letters only if its row says PASS on your deployment.
"""

from __future__ import annotations

import math
import sys

import torch

from typed_decisions.backend_impls.transformers import TransformersBackend
from typed_decisions.client import Calibration, Decider
from typed_decisions.prompt import render_prefix
from typed_decisions.types import ChoiceQuestion, NoulQuestion, ScoreQuestion

STATES = [
    {"ticket": "I was charged twice for my subscription this month. Please refund one."},
    {"ticket": "The dashboard has been down for all users since 9am, we are losing sales."},
    {"ticket": "Can you send pricing for 200 seats on the enterprise plan?"},
]
QUESTIONS = {
    "route": ChoiceQuestion("Which team should handle this?", ["billing", "technical", "sales"]),
    "urgent": NoulQuestion("Is this urgent?"),
    "severity": ScoreQuestion("Rate severity.", {"low": 1.0, "medium": 2.0, "high": 3.0}),
    "team_by_text": ChoiceQuestion("Which team should handle this?",
                                   ["billing", "technical support", "sales"]),
}


def softmax(xs):
    top = max(xs)
    e = [math.exp(x - top) for x in xs]
    return [v / sum(e) for v in e]


def main() -> None:
    kind, model, url = sys.argv[1], sys.argv[2], sys.argv[3]
    reference = TransformersBackend.from_pretrained(model, device="cpu", dtype=torch.float32)
    if kind == "vllm":
        from typed_decisions.backend_impls.vllm import VLLMBackend
        remotes = {"vllm default (exact)": VLLMBackend(model, reference.tokenizer, base_url=url),
                   "vllm cache_letters=True": VLLMBackend(model, reference.tokenizer,
                                                          base_url=url, cache_letters=True)}
    else:
        from typed_decisions.backend_impls.sglang import SGLangBackend
        remotes = {"sglang": SGLangBackend(reference.tokenizer, base_url=url)}

    # Force the last question onto sequence scoring, so both paths are exercised.
    planner = Decider(reference, calibration=Calibration(), sequence_threshold=26)
    plans, branches = planner._plan({k: v for k, v in QUESTIONS.items() if k != "team_by_text"})
    seq_planner = Decider(reference, sequence_threshold=0)
    _, seq_branches = seq_planner._plan({"team_by_text": QUESTIONS["team_by_text"]})
    branches = branches + seq_branches
    names = [p.name for p in plans] + ["team_by_text (sequence)"]

    for label, remote in remotes.items():
        worst = 0.0
        for state in STATES:
            prefix = render_prefix(state)
            ref_rows = reference.score(prefix, branches)
            for _ in range(2):   # twice: the second pass is where cache hits happen
                rem_rows = remote.score(prefix, branches)
                for name, a, b in zip(names, ref_rows, rem_rows):
                    worst = max(worst, max(abs(x - y) for x, y in zip(softmax(a), softmax(b))))
        stats = remote.stats
        verdict = "PASS" if worst < 0.02 else "FAIL: disagrees with the reference"
        print(f"{label:26} worst |dp| {worst:.4f}   prefix cache {stats.cached_tokens}/"
              f"{stats.prompt_tokens} ({stats.hit_rate:.1%})   {verdict}")


if __name__ == "__main__":
    main()
