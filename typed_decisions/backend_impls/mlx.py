"""Apple-silicon backend on MLX: the same forked scoring as TransformersBackend, natively.

Why a second local backend: on a Mac, PyTorch's MPS path runs Qwen3.5's linear-attention
layers through slow reference kernels, and a 9B model in bfloat16 (~18 GB) does not fit
beside anything else in 24 GB of unified memory -- it swapped and never finished. MLX has
native kernels for these layers and loads 4-/8-bit quantised weights (a 9B in ~5 GB).

Scoring is identical in shape to TransformersBackend: prefill the state once into a prompt
cache, fork the cache per question branch (and per option for option-text scoring), read
logits at the answer position. `score_uncached` / `score_sequence_uncached` run each prompt
whole, and tests assert the fork changes nothing.

Quantised weights are a different model from the bf16 one: compare results, don't assume
them (examples/verify_backend.py-style checks apply).
"""

from __future__ import annotations

import copy
from typing import Sequence

import mlx.core as mx

from ..backends import Branch, PrefixStats, continuation_ids, shared_prefix_length


class _Tokenizer:
    """encode() without special tokens, as the label allocator and prefix logic expect."""

    def __init__(self, tokenizer):
        self._tok = tokenizer

    def encode(self, text: str) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=False))

    def __getattr__(self, name):
        return getattr(self._tok, name)


class MLXBackend:
    def __init__(self, model, tokenizer, tolerance: int = 1):
        self.model = model
        self.tokenizer = _Tokenizer(tokenizer)
        self.tolerance = tolerance
        self.stats = PrefixStats()

    @classmethod
    def from_pretrained(cls, name: str):
        from mlx_lm import load
        model, tokenizer = load(name)
        return cls(model, tokenizer)

    # -- cache plumbing ---------------------------------------------------------------

    def _new_cache(self):
        from mlx_lm.models.cache import make_prompt_cache
        return make_prompt_cache(self.model)

    def _run(self, ids: Sequence[int], cache):
        """Feed `ids` through the model, extending `cache` in place; returns logits."""
        logits = self.model(mx.array([list(ids)]), cache=cache)
        mx.eval(logits)
        return logits

    def _fork(self, cache):
        # Cache objects hold mx.arrays; MLX arrays are immutable values, so a deep copy of
        # the container objects is a true fork (the prefill is not recomputed).
        return copy.deepcopy(cache)

    # -- scoring ----------------------------------------------------------------------

    def encode_branches(self, prefix: str, branches: Sequence[Branch]):
        prefix_ids = self.tokenizer.encode(prefix)
        wholes = [self.tokenizer.encode(prefix + b.text) for b in branches]
        shared = shared_prefix_length(prefix_ids, wholes, self.tolerance)
        return prefix_ids[:shared], [w[shared:] for w in wholes]

    def score(self, prefix: str, branches: Sequence[Branch]) -> list[list[float]]:
        if not branches:
            return []
        shared_ids, branch_id_lists = self.encode_branches(prefix, branches)
        base = self._new_cache()
        self._run(shared_ids, base)

        rows = []
        for branch, branch_ids in zip(branches, branch_id_lists):
            cache = self._fork(base)
            logits = self._run(branch_ids, cache)
            if branch.continuations:
                rows.append(self._sequences(logits, cache, shared_ids + branch_ids,
                                            prefix + branch.text, branch))
            else:
                last = logits[0, -1]
                rows.append([float(last[t]) for t in branch.token_ids])
        self.stats.add(
            prompt_tokens=sum(len(shared_ids) + len(ids) for ids in branch_id_lists),
            cached_tokens=len(shared_ids) * (len(branches) - 1),
        )
        return rows

    def _sequences(self, logits, cache, context_ids, context, branch) -> list[float]:
        first = logits[0, -1].astype(mx.float32)
        first = first - mx.logsumexp(first)
        scores = []
        for option in branch.continuations:
            known, ids = continuation_ids(self.tokenizer, context, option)
            if known != context_ids:
                raise ValueError("context re-encodes differently; cannot fork option scoring")
            total = float(first[ids[0]])
            if len(ids) > 1:
                rest = self._run(ids[:-1], self._fork(cache))[0].astype(mx.float32)
                rest = rest - mx.logsumexp(rest, axis=-1, keepdims=True)
                total += sum(float(rest[i, t]) for i, t in enumerate(ids[1:]))
            scores.append(total / len(ids) if branch.normalize == "mean" else total)
        return scores

    def score_uncached(self, prefix: str, branch: Branch) -> list[float]:
        """Reference path: one full forward over prefix+branch, no cache reuse."""
        logits = self._run(self.tokenizer.encode(prefix + branch.text), self._new_cache())
        last = logits[0, -1]
        return [float(last[t]) for t in branch.token_ids]

    def score_sequence_uncached(self, prefix: str, branch: Branch) -> list[float]:
        context = prefix + branch.text
        context_ids = self.tokenizer.encode(context)
        scores = []
        for option in branch.continuations:
            _, ids = continuation_ids(self.tokenizer, context, option)
            logits = self._run(context_ids + ids, self._new_cache())[0].astype(mx.float32)
            logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            start = len(context_ids) - 1
            total = sum(float(logp[start + i, t]) for i, t in enumerate(ids))
            scores.append(total / len(ids) if branch.normalize == "mean" else total)
        return scores
