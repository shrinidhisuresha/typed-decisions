"""The seam between prompt construction and whatever actually runs the model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .labels import Tokenizer


@dataclass(frozen=True)
class Branch:
    """One question's continuation off the shared prefix, plus the labels to read.

    token_ids restrict sampling; rendered ("` A`", "` B`", ...) is how HTTP backends
    key the returned logprob table, which is string-keyed rather than id-keyed.
    """

    text: str
    token_ids: list[int]
    rendered: list[str]
    # Sequence scoring (design doc section 2, for > 26 options): when set, each option
    # is its own continuation after `text`, and the backend returns the MEAN log-prob of
    # each continuation's tokens instead of a label logit. token_ids/rendered are unused.
    continuations: tuple[str, ...] = ()
    # "mean": mean token log-prob (length-normalised). "sum": total log-prob; with
    # contextual calibration this becomes PMI, log p(option|state) - log p(option|no state),
    # so a predictable multi-token tail ("shipping AND LOGISTICS") cancels out.
    normalize: str = "mean"

    @property
    def n_options(self) -> int:
        return len(self.continuations) if self.continuations else len(self.token_ids)


@dataclass
class PrefixStats:
    """Running count of prompt tokens scored vs. prompt tokens served from a KV cache.

    The Phase 2 acceptance number (design doc section 7): if hit_rate is not close to
    (k-1)/k for k questions over a long state, the prefix is not really being shared.
    """

    prompt_tokens: int = 0
    cached_tokens: int = 0

    def add(self, prompt_tokens: int, cached_tokens: int) -> None:
        self.prompt_tokens += prompt_tokens
        self.cached_tokens += cached_tokens

    @property
    def hit_rate(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0


class SeamError(ValueError):
    """An option merges with the prompt across the seam, so its tokens cannot be scored
    as a clean continuation of the question."""


def continuation_ids(tokenizer: Tokenizer, context: str, option: str) -> tuple[list[int], list[int]]:
    """(context ids, option ids), the option's tokens sliced IN CONTEXT off the joint
    encoding -- never encoded standalone (see README, the token-level prefix invariant)."""
    context_ids = tokenizer.encode(context)
    full = tokenizer.encode(context + option)
    if full[: len(context_ids)] != context_ids:
        raise SeamError(f"option {option!r} merges with the prompt across the seam")
    ids = full[len(context_ids):]
    if not ids:
        raise SeamError(f"option {option!r} encodes to no tokens")
    return context_ids, ids


class PrefixError(SeamError):
    """The state is not a token-level prefix of the full prompt, so its KV cache
    cannot legally be shared across branches."""


def _match_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def shared_prefix_length(
    prefix_ids: Sequence[int],
    whole_ids: Sequence[Sequence[int]],
    tolerance: int = 1,
) -> int:
    r"""How many leading tokens every branch genuinely shares with the state.

    Not simply len(prefix_ids): tokenizers merge across the seam. If the state ends
    with "\n" and a branch starts with "\n", BPE emits one "\n\n" token in context,
    so the state's own final token never appears in the real sequence. We share
    everything up to that straddling token and let each branch re-encode it.

    Losing more than `tolerance` tokens means the state is not really at the front of
    the prompt -- a broken template, which silently destroys cache sharing.
    """
    capped = min(_match_len(prefix_ids, whole) for whole in whole_ids)
    shared = min(capped, len(prefix_ids))
    if shared < len(prefix_ids) - tolerance:
        raise PrefixError(
            f"state is not a token-level prefix of the rendered prompt "
            f"(shared {shared} of {len(prefix_ids)} tokens); check that render_prefix() "
            "output really leads every branch"
        )
    return shared


class Backend(Protocol):
    tokenizer: Tokenizer

    def score(self, prefix: str, branches: Sequence[Branch]) -> list[list[float]]:
        """Return the logit for each of branch.token_ids at the position after
        prefix + branch.text -- one list per branch, aligned to token_ids. For a branch
        with `continuations`, return each continuation's mean token log-prob instead.

        Implementations must prefill `prefix` once and fork, never re-prefill per branch.
        """
        ...
