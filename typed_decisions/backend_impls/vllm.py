"""vLLM backend, over the OpenAI-compatible HTTP server.

Prefix sharing here is delegated to the server: vLLM's automatic prefix caching
recognises that every branch starts with the same state and reuses its KV blocks.
That means you MUST start the server with prefix caching enabled, and you must not
put anything question-specific before the state.

Uses /v1/completions, never /v1/chat/completions -- a chat template would wrap the
prompt and break the shared-prefix invariant.

Phase 2: the state is prefilled once on its own (`warm_prefix`), then letter branches go out
batched -- one request per label set -- so each lands on an already-cached state.

By default every option -- letters included -- is scored from prompt logprobs, which is
exact but bypasses the prefix cache. `cache_letters=True` scores letters at the generated
position instead, riding the cache; it needs the server started with
    --enable-prefix-caching --logprobs-mode processed_logprobs --enable-prompt-tokens-details
and must be verified per deployment (see __init__ and examples/verify_backend.py). Start the server
with `--enable-prompt-tokens-details` and `stats` will carry vLLM's own cached-token
count, so the hit rate is measured rather than assumed.
"""

from __future__ import annotations

from typing import Sequence

import httpx

from ..backends import Branch, PrefixStats, continuation_ids
from ..labels import Tokenizer


class VLLMBackend:
    def __init__(
        self,
        model: str,
        tokenizer: Tokenizer,
        base_url: str = "http://localhost:8000",
        client: httpx.Client | None = None,
        timeout: float = 60.0,
        warm_prefix: bool = True,
        cache_letters: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.client = client or httpx.Client(base_url=base_url, timeout=timeout)
        self.warm_prefix = warm_prefix
        # Off by default: on vLLM 0.11.0 (CPU) a prefix-cache HIT changed label logprobs
        # (0.1977 uncached, 0.2121 cached, same request; the reference model says 0.1977).
        # Turn on only after examples/verify_backend.py passes in cached mode on YOUR
        # deployment. Off, every option is scored via prompt logprobs, which bypass the
        # cache: exact, but it forgoes the Phase 2 saving.
        self.cache_letters = cache_letters
        self.stats = PrefixStats()

    def score(self, prefix: str, branches: Sequence[Branch]) -> list[list[float]]:
        if not branches:
            return []
        if not self.cache_letters:
            return self._sequences(prefix, branches)
        if self.warm_prefix:
            self._post({"model": self.model, "prompt": prefix, "max_tokens": 1,
                        "temperature": 0.0})
        rows: list[list[float] | None] = [None] * len(branches)
        groups: dict[tuple[int, ...], list[int]] = {}
        for i, branch in enumerate(branches):
            if not branch.continuations:
                groups.setdefault(tuple(branch.token_ids), []).append(i)
        for token_ids, members in groups.items():
            for i, row in zip(members, self._letters(prefix, [branches[i] for i in members])):
                rows[i] = row
        sequences = [i for i, b in enumerate(branches) if b.continuations]
        if sequences:
            for i, row in zip(sequences, self._sequences(prefix, [branches[i] for i in sequences])):
                rows[i] = row
        return rows  # type: ignore[return-value]

    def _letters(self, prefix: str, branches: Sequence[Branch]) -> list[list[float]]:
        """Label logprobs at the ONE generated position, keyed by token id. Cheap, and it
        rides the prefix cache (no prompt logprobs).

        Needs the server started with `--logprobs-mode processed_logprobs`, and
        temperature 0: vLLM's default raw logprobs are taken BEFORE allowed_token_ids
        masks the vocabulary, so the top-k is not the labels. Verified against a live
        server with examples/verify_backend.py; a misconfigured server is detected below.
        """
        token_ids = list(branches[0].token_ids)
        body = {
            "model": self.model,
            "prompt": [prefix + b.text for b in branches],
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": len(token_ids),
            "allowed_token_ids": token_ids,
            "return_tokens_as_token_ids": True,
        }
        choices = sorted(self._post(body)["choices"], key=lambda c: c.get("index", 0))
        if len(choices) != len(branches):
            raise RuntimeError(f"expected {len(branches)} choices, got {len(choices)}")
        rows = []
        for choice in choices:
            top = choice["logprobs"]["top_logprobs"][0]
            by_id = {int(str(k).split(":")[-1]): float(v) for k, v in top.items()}
            if set(by_id) != set(token_ids):
                raise RuntimeError(
                    f"server returned logprobs for token ids {sorted(by_id)}, not the labels "
                    f"{sorted(token_ids)}: start vLLM with --logprobs-mode processed_logprobs")
            rows.append([by_id[t] for t in token_ids])
        return rows

    def _options(self, context: str, branch: Branch) -> list[list[int]]:
        """Each option's in-context token ids. A letter branch is scored as one-token
        continuations (" A", " B", ...), checked against the allocated label ids."""
        if branch.continuations:
            return [continuation_ids(self.tokenizer, context, o)[1] for o in branch.continuations]
        ids = [continuation_ids(self.tokenizer, context, r)[1] for r in branch.rendered]
        if ids != [[t] for t in branch.token_ids]:
            raise RuntimeError(f"labels {branch.rendered} do not encode to their allocated "
                               f"single tokens {branch.token_ids} after this prompt")
        return ids

    def _sequences(self, prefix: str, branches: Sequence[Branch]) -> list[list[float]]:
        """Every option of every branch in ONE request: echo the prompt with its logprobs,
        read the option's tokens back from the end of the prompt, check the ids.

        Asking for prompt logprobs makes vLLM skip its prefix cache for these requests and
        compute full-vocabulary logprobs at every prompt position, so this is much more
        expensive than the letter path on a long state. It is correct, not cheap.
        """
        jobs = []
        for bi, branch in enumerate(branches):
            context = prefix + branch.text
            for option_text, ids in zip(branch.continuations or branch.rendered,
                                        self._options(context, branch)):
                jobs.append((bi, ids, context + option_text, branch.normalize))
        body = {
            "model": self.model,
            "prompt": [text for _, _, text, _ in jobs],
            "max_tokens": 1,
            "temperature": 0.0,
            "echo": True,
            "logprobs": 1,
            "return_tokens_as_token_ids": True,
        }
        choices = sorted(self._post(body)["choices"], key=lambda c: c.get("index", 0))
        if len(choices) != len(jobs):
            raise RuntimeError(f"expected {len(jobs)} choices, got {len(choices)}")
        rows: list[list[float]] = [[] for _ in branches]
        for (bi, ids, _, how), choice in zip(jobs, choices):
            lp = choice["logprobs"]
            prompt_tokens = lp["tokens"][:-1]           # last one is the generated token
            prompt_logprobs = lp["token_logprobs"][:-1]
            got = [int(str(t).split(":")[-1]) for t in prompt_tokens[-len(ids):]]
            picked = prompt_logprobs[-len(ids):]
            if got != ids or any(x is None for x in picked):
                raise RuntimeError(f"server tokens {got} do not match the option's in-context "
                                   f"tokens {ids}; tokenizer mismatch between client and server")
            total = sum(float(x) for x in picked)
            rows[bi].append(total / len(ids) if how == "mean" else total)
        return rows

    def _post(self, body: dict) -> dict:
        response = self.client.post("/v1/completions", json=body)
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        self.stats.add(int(usage.get("prompt_tokens", 0)), int(details.get("cached_tokens") or 0))
        return payload
