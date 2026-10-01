"""SGLang backend, over the native /generate endpoint (design doc section 5, Phase 2).

SGLang's RadixAttention is built for exactly this workload: one long state, many
short branches forked off it. Two things make that work here:

1. **Warm, then fan out.** The state is prefilled once on its own, so it sits in the
   radix tree before any branch arrives. All branches then go in ONE batched request
   and every one of them hits the cached state.
2. **Ask for the label ids, not the top-k.** `token_ids_logprob` returns the logprob
   of each allocated label token by id, so an option can never fall out of a top-k
   table the way it can over the OpenAI-compatible endpoint.

`stats` accumulates the server's own `cached_tokens` report -- that is the cache hit
rate Phase 2 asks you to verify, measured by the server rather than assumed.

Start the server with the radix cache on (the default; do not pass
`--disable-radix-cache`).
"""

from __future__ import annotations

from typing import Sequence

import httpx

from ..backends import Branch, PrefixStats, continuation_ids
from ..labels import Tokenizer

# temperature=1.0 is deliberate. Only the distribution at the next position is read,
# never the sampled token, and at temperature 1.0 the returned logprobs are the
# model's raw log-softmax -- no greedy/temperature post-processing can leak in.
_SAMPLING = {"max_new_tokens": 1, "temperature": 1.0}


class SGLangBackend:
    def __init__(
        self,
        tokenizer: Tokenizer,
        base_url: str = "http://localhost:30000",
        client: httpx.Client | None = None,
        timeout: float = 60.0,
        warm_prefix: bool = True,
    ):
        self.tokenizer = tokenizer
        self.client = client or httpx.Client(base_url=base_url, timeout=timeout)
        self.warm_prefix = warm_prefix
        self.stats = PrefixStats()

    def score(self, prefix: str, branches: Sequence[Branch]) -> list[list[float]]:
        if not branches:
            return []
        if self.warm_prefix:
            self._record(self._post({"text": prefix, "sampling_params": _SAMPLING}))

        rows: list[list[float] | None] = [None] * len(branches)
        letters = [i for i, b in enumerate(branches) if not b.continuations]
        if letters:
            body = {
                "text": [prefix + branches[i].text for i in letters],
                "sampling_params": _SAMPLING,
                "return_logprob": True,
                "logprob_start_len": -1,
                "token_ids_logprob": [list(branches[i].token_ids) for i in letters],
            }
            for i, output in zip(letters, self._batch(body, len(letters))):
                self._record(output)
                rows[i] = self._read(output, branches[i])
        sequences = [i for i, b in enumerate(branches) if b.continuations]
        if sequences:
            for i, row in zip(sequences, self._sequences(prefix, [branches[i] for i in sequences])):
                rows[i] = row
        return rows  # type: ignore[return-value]

    def _batch(self, body: dict, n: int) -> list:
        outputs = self._post(body)
        if not isinstance(outputs, list) or len(outputs) != n:
            raise RuntimeError(
                f"expected {n} outputs from a batched /generate, got "
                f"{len(outputs) if isinstance(outputs, list) else type(outputs).__name__}"
            )
        return outputs

    def _sequences(self, prefix: str, branches: Sequence[Branch]) -> list[list[float]]:
        """Every option of every sequence branch in ONE batched request. The prompt is
        context + option; the server returns the input-token logprobs from just before the
        option, and the option's own tokens are read back from the end and id-checked."""
        jobs = []   # (branch index, option ids, text, start)
        for bi, branch in enumerate(branches):
            context = prefix + branch.text
            for option in branch.continuations:
                context_ids, ids = continuation_ids(self.tokenizer, context, option)
                jobs.append((bi, ids, context + option, max(len(context_ids) - 1, 0),
                             branch.normalize))
        body = {
            "text": [text for _, _, text, _, _ in jobs],
            "sampling_params": _SAMPLING,
            "return_logprob": True,
            "logprob_start_len": [start for _, _, _, start, _ in jobs],
        }
        rows: list[list[float]] = [[] for _ in branches]
        for (bi, ids, _, _, how), output in zip(jobs, self._batch(body, len(jobs))):
            self._record(output)
            entries = (output.get("meta_info", {}).get("input_token_logprobs") or [])[-len(ids):]
            got = [int(e[1]) for e in entries]
            if got != ids or any(e[0] is None for e in entries):
                raise RuntimeError(f"server tokens {got} do not match the option's in-context "
                                   f"tokens {ids}; tokenizer mismatch between client and server")
            total = sum(float(e[0]) for e in entries)
            rows[bi].append(total / len(ids) if how == "mean" else total)
        return rows

    def _post(self, body: dict):
        response = self.client.post("/generate", json=body)
        response.raise_for_status()
        return response.json()

    def _record(self, output: dict) -> None:
        meta = output.get("meta_info", {})
        self.stats.add(int(meta.get("prompt_tokens", 0)), int(meta.get("cached_tokens", 0)))

    @staticmethod
    def _read(output: dict, branch: Branch) -> list[float]:
        # output_token_ids_logprobs: one entry per generated position, each a list of
        # (logprob, token_id, text) for the requested ids. We generate one position.
        positions = output.get("meta_info", {}).get("output_token_ids_logprobs") or []
        if not positions:
            raise RuntimeError(
                "server returned no output_token_ids_logprobs; this SGLang build may "
                "not support token_ids_logprob"
            )
        table = {int(entry[1]): float(entry[0]) for entry in positions[0]}
        missing = [t for t in branch.token_ids if t not in table]
        if missing:
            raise RuntimeError(f"server did not return logprobs for token ids {missing!r}")
        return [table[t] for t in branch.token_ids]
