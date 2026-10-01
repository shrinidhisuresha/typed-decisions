"""Retrieve, then score: Choice over many plausible options (design doc section 3C).

Scoring every option by its text breaks down when many options are plausible: at 40
options (4 true routes + 36 unrelated departments) the 4B model reached ~0.35 accuracy
whatever the scoring rule. A bi-encoder is good at the coarse cut and cheap enough to run
over every option; the decoder is good at the fine decision among a few. So:

    embed state and options -> keep the top k -> score those k with letters,
    with permutation and contextual calibration as usual

Options outside the shortlist get probability 0. The answer's distribution is therefore
CONDITIONAL on the shortlist, and recall@k -- how often the right option survives the cut
-- is a hard ceiling on accuracy. Measure it (examples/high_cardinality.py does).

Embedding conventions follow the Qwen3-Embedding model card: last-token pooling, left
padding, L2-normalised vectors, and an instruction prefix on the query only.
"""

from __future__ import annotations

from typing import Protocol, Sequence


class Retriever(Protocol):
    def rank(self, query: str, options: Sequence[str], task: str) -> list[float]:
        """A relevance score per option, aligned to `options` (higher is better)."""
        ...


def shortlist(scores: Sequence[float], k: int) -> list[int]:
    """Indices of the top-k scores, returned in their ORIGINAL order, so the listing the
    decoder sees is the caller's declared order (permutation handles position bias)."""
    if k < 1:
        raise ValueError("shortlist size must be at least 1")
    top = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
    return sorted(top)


class EmbeddingRetriever:
    DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"

    def __init__(self, model: str = DEFAULT_MODEL, device: str | None = None,
                 max_length: int = 1024, batch_size: int = 32):
        # Loaded on first use, not here: most requests never have a Choice over 26 options,
        # and loading eagerly doubled a CPU server's start-up memory (OOM in an 8 GB container).
        self.name = model
        self._device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self._option_cache: dict[str, object] = {}   # option text -> vector
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        from .backend_impls.transformers import best_device

        self.torch = torch
        self.device = best_device(self._device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name, padding_side="left")
        dtype = torch.float32 if self.device == "cpu" else torch.float16
        self.model = AutoModel.from_pretrained(self.name, dtype=dtype).to(self.device).eval()
        self._loaded = True

    def _embed(self, texts: Sequence[str]):
        self._load()
        torch = self.torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(list(texts[i:i + self.batch_size]), padding=True,
                                       truncation=True, max_length=self.max_length,
                                       return_tensors="pt").to(self.device)
                hidden = self.model(**batch).last_hidden_state
                # left padding: the last position is every sequence's final real token
                vectors = hidden[:, -1].float()
                out.append(torch.nn.functional.normalize(vectors, p=2, dim=1).cpu())
        return torch.cat(out)

    def rank(self, query: str, options: Sequence[str], task: str) -> list[float]:
        missing = [o for o in options if o not in self._option_cache]
        if missing:
            for option, vector in zip(missing, self._embed(missing)):
                self._option_cache[option] = vector
        q = self._embed([f"Instruct: {task}\nQuery:{query}"])[0]
        return [float(q @ self._option_cache[o]) for o in options]
