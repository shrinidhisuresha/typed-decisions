"""Local HuggingFace backend. Prefills the state once, then forks the KV cache per question.

This is the reference implementation of the shared-prefix idea: unlike the HTTP
backend, the fork is explicit here rather than delegated to a server's cache.
"""

from __future__ import annotations

import copy
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..backends import (  # noqa: F401  (PrefixError/shared_prefix_length re-exported)
    Branch,
    PrefixError,
    PrefixStats,
    continuation_ids,
    shared_prefix_length,
)


def best_device(explicit: str | None = None) -> str:
    """Pick the fastest available device unless the caller named one.

    Not a micro-optimisation: on Apple Silicon this model runs ~50x faster on MPS
    than on CPU, because the CPU path falls back to unoptimised kernels.
    """
    if explicit is not None:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _HFTokenizer:
    """Adapts a HF tokenizer to the minimal encode() the label allocator needs."""

    def __init__(self, tokenizer):
        self._tok = tokenizer

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def __getattr__(self, name):
        return getattr(self._tok, name)


class TransformersBackend:
    def __init__(self, model, tokenizer, device: str = "cpu", tolerance: int = 1):
        self.model = model
        self.tokenizer = _HFTokenizer(tokenizer)
        self.device = device
        self.tolerance = tolerance
        self.stats = PrefixStats()

    @classmethod
    def from_pretrained(cls, name: str, device: str | None = None, dtype=None):
        resolved = best_device(device)
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(
            name, dtype=dtype if dtype is not None else torch.float32
        )
        model.to(resolved).eval()
        return cls(model, tokenizer, resolved)

    def _tensor(self, ids: list[int]) -> torch.Tensor:
        return torch.tensor([ids], device=self.device)

    def encode_branches(self, prefix: str, branches: Sequence[Branch]):
        """Tokenise every full prompt in context, then split at the shared boundary."""
        prefix_ids = self.tokenizer.encode(prefix)
        wholes = [self.tokenizer.encode(prefix + b.text) for b in branches]
        shared = shared_prefix_length(prefix_ids, wholes, self.tolerance)
        return prefix_ids[:shared], [w[shared:] for w in wholes]

    @torch.inference_mode()
    def score(self, prefix: str, branches: Sequence[Branch]) -> list[list[float]]:
        shared_ids, branch_id_lists = self.encode_branches(prefix, branches)
        prefix_len = len(shared_ids)

        prefilled = self.model(input_ids=self._tensor(shared_ids), use_cache=True)

        rows = []
        for branch, branch_ids in zip(branches, branch_id_lists):
            out = self._extend(prefilled.past_key_values, prefix_len, branch_ids)
            if branch.continuations:
                whole = shared_ids + branch_ids
                rows.append(self._sequences(out, whole, prefix + branch.text, branch))
            else:
                rows.append(self._read(out.logits, branch))
        # The first branch pays for the prefill; every other branch reads it from cache.
        self.stats.add(
            prompt_tokens=sum(prefix_len + len(ids) for ids in branch_id_lists),
            cached_tokens=prefix_len * (len(branches) - 1),
        )
        return rows

    def _extend(self, cache, start: int, ids: Sequence[int]):
        """Run `ids` on a COPY of `cache` (length `start`); the original stays reusable."""
        total = start + len(ids)
        return self.model(
            input_ids=self._tensor(list(ids)),
            past_key_values=copy.deepcopy(cache),
            attention_mask=torch.ones((1, total), dtype=torch.long, device=self.device),
            cache_position=torch.arange(start, total, device=self.device),
            use_cache=True,
        )

    def continuation_ids(self, context_ids: list[int], context: str, option: str) -> list[int]:
        known, ids = continuation_ids(self.tokenizer, context, option)
        if known != context_ids:
            raise PrefixError("context re-encodes differently; cannot fork option scoring")
        return ids

    def _sequences(self, out, context_ids: list[int], context: str, branch: Branch) -> list[float]:
        """Mean log-prob of each continuation, forked off the question's cache."""
        context_len = len(context_ids)
        first = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        scores = []
        for option in branch.continuations:
            ids = self.continuation_ids(context_ids, context, option)
            total = float(first[ids[0]])
            if len(ids) > 1:
                rest = self._extend(out.past_key_values, context_len, ids[:-1])
                logp = torch.log_softmax(rest.logits[0].float(), dim=-1)
                total += float(sum(logp[i, t] for i, t in enumerate(ids[1:])))
            scores.append(total / len(ids) if branch.normalize == "mean" else total)
        return scores

    @torch.inference_mode()
    def score_sequence_uncached(self, prefix: str, branch: Branch) -> list[float]:
        """Naive path: one full forward per option. Reference for tests."""
        context_ids = self.tokenizer.encode(prefix + branch.text)
        scores = []
        for option in branch.continuations:
            ids = self.continuation_ids(context_ids, prefix + branch.text, option)
            out = self.model(input_ids=self._tensor(context_ids + ids))
            logp = torch.log_softmax(out.logits[0].float(), dim=-1)
            start = len(context_ids) - 1
            total = float(sum(logp[start + i, t] for i, t in enumerate(ids)))
            scores.append(total / len(ids) if branch.normalize == "mean" else total)
        return scores

    @torch.inference_mode()
    def score_uncached(self, prefix: str, branch: Branch) -> list[float]:
        """Naive path: one full forward over prefix+branch. Reference for tests."""
        out = self.model(input_ids=self._tensor(self.tokenizer.encode(prefix + branch.text)))
        return self._read(out.logits, branch)

    @staticmethod
    def _read(logits: torch.Tensor, branch: Branch) -> list[float]:
        last = logits[0, -1]
        return [float(last[token_id]) for token_id in branch.token_ids]
