import json
import re

import httpx
import pytest

from typed_decisions.backends import Branch
from typed_decisions.backend_impls.vllm import VLLMBackend


class WordTokenizer:
    """One token per word (leading space included), ids stable per session. Unlike
    FakeTokenizer it tokenises a label after a prompt the same way as on its own."""

    def __init__(self):
        self.ids: dict[str, int] = {}

    def encode(self, text):
        return [self.ids.setdefault(t, 100 + len(self.ids)) for t in re.findall(r" ?\S+|\s", text)]


TOK = WordTokenizer()
A, B = TOK.encode(" A")[0], TOK.encode(" B")[0]
PREFIX = "<state>\nticket: charged twice\n</state>\n"
BRANCH = Branch("\nPick one.\nA) billing\nB) sales\nAnswer:", [A, B], [" A", " B"])
OTHER = Branch("\nUrgent?\nA) yes\nB) no\nAnswer:", [A, B], [" A", " B"])
LOGPROB = {A: -0.5, B: -1.5}


def server(captured, raw=False, bad_ids=False, cached_tokens=None):
    def handler(request):
        body = json.loads(request.content)
        captured.append(body)
        prompts = body["prompt"] if isinstance(body["prompt"], list) else [body["prompt"]]
        choices = []
        for i, text in enumerate(prompts):
            if body.get("echo"):
                ids = TOK.encode(text)
                if bad_ids:
                    ids = [t + 1 for t in ids]
                logprobs = {"tokens": [f"token_id:{t}" for t in ids] + ["token_id:1"],
                            "token_logprobs": [None] + [LOGPROB.get(t, -2.0) for t in ids[1:]] + [-0.1]}
            elif "logprobs" in body:
                allowed = body["allowed_token_ids"]
                top = {f"token_id:{t}": LOGPROB[t] for t in allowed}
                if raw:   # vLLM's default raw mode: top-k over the whole vocabulary
                    top = {"token_id:5": -0.1, f"token_id:{allowed[0]}": -3.0}
                logprobs = {"top_logprobs": [top]}
            else:
                logprobs = None
            choices.append({"index": i, "logprobs": logprobs})
        usage = {"prompt_tokens": 10 * len(prompts)}
        if cached_tokens is not None:
            usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
        return httpx.Response(200, json={"choices": choices, "usage": usage})
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://vllm.test")


def backend(captured, **kwargs):
    client_kwargs = {k: kwargs.pop(k) for k in ("raw", "bad_ids", "cached_tokens") if k in kwargs}
    return VLLMBackend(model="m", tokenizer=TOK, client=server(captured, **client_kwargs), **kwargs)


def scoring(captured):
    return [b for b in captured if "logprobs" in b]


# --- default: exact, every option from prompt logprobs -------------------------------

def test_uses_the_completions_endpoint_not_chat():
    captured = []
    backend(captured).score(PREFIX, [BRANCH])
    assert all("prompt" in b and "messages" not in b for b in captured)


def test_default_scores_letters_as_one_token_continuations_in_one_request():
    captured = []
    rows = backend(captured).score(PREFIX, [BRANCH, OTHER])
    assert rows == [[-0.5, -1.5], [-0.5, -1.5]]
    assert len(captured) == 1 and captured[0]["echo"] is True
    assert captured[0]["prompt"] == [PREFIX + BRANCH.text + " A", PREFIX + BRANCH.text + " B",
                                     PREFIX + OTHER.text + " A", PREFIX + OTHER.text + " B"]


def test_default_does_not_warm_or_use_allowed_token_ids():
    captured = []
    backend(captured).score(PREFIX, [BRANCH])
    assert "allowed_token_ids" not in captured[0] and len(captured) == 1


def test_every_prompt_starts_with_the_state():
    captured = []
    backend(captured).score(PREFIX, [BRANCH, OTHER])
    assert all(p.startswith(PREFIX) for p in captured[0]["prompt"])


def test_generates_exactly_one_token_position():
    captured = []
    backend(captured).score(PREFIX, [BRANCH])
    assert all(b["max_tokens"] == 1 for b in captured)


def test_a_client_server_tokenizer_mismatch_is_refused():
    with pytest.raises(RuntimeError, match="tokenizer mismatch"):
        backend([], bad_ids=True).score(PREFIX, [BRANCH])


def test_a_label_that_does_not_encode_to_its_allocated_token_is_refused():
    wrong = Branch(BRANCH.text, [A, 999], [" A", " B"])
    with pytest.raises(RuntimeError, match="allocated"):
        backend([]).score(PREFIX, [wrong])


def test_records_the_servers_cached_token_count():
    b = backend([], cached_tokens=0)
    b.score(PREFIX, [BRANCH])
    assert b.stats.prompt_tokens == 20 and b.stats.cached_tokens == 0


# --- cache_letters=True: the generated position, riding the prefix cache -------------

def test_cache_mode_warms_the_bare_state_first():
    captured = []
    backend(captured, cache_letters=True).score(PREFIX, [BRANCH])
    assert captured[0]["prompt"] == PREFIX and "logprobs" not in captured[0]


def test_cache_mode_warm_up_can_be_turned_off():
    captured = []
    backend(captured, cache_letters=True, warm_prefix=False).score(PREFIX, [BRANCH])
    assert len(captured) == 1 and "allowed_token_ids" in captured[0]


def test_cache_mode_batches_a_label_set_and_keys_logprobs_by_token_id():
    captured = []
    branches = [Branch(f"\nQ{i}?\nA) x\nB) y\nAnswer:", [A, B], [" A", " B"]) for i in range(20)]
    rows = backend(captured, cache_letters=True).score(PREFIX, branches)
    assert rows == [[-0.5, -1.5]] * 20
    body = scoring(captured)[0]
    assert len(scoring(captured)) == 1 and len(body["prompt"]) == 20
    assert body["allowed_token_ids"] == [A, B] and body["return_tokens_as_token_ids"] is True
    assert body["temperature"] == 0.0


def test_cache_mode_detects_a_server_without_processed_logprobs():
    with pytest.raises(RuntimeError, match="processed_logprobs"):
        backend([], cache_letters=True, raw=True).score(PREFIX, [BRANCH])


def test_cache_mode_splits_by_label_set_and_keeps_order():
    C = TOK.encode(" C")[0]
    LOGPROB[C] = -2.5
    three = Branch("\nPick.\nA) x\nB) y\nC) z\nAnswer:", [A, B, C], [" A", " B", " C"])
    captured = []
    rows = backend(captured, cache_letters=True).score(PREFIX, [BRANCH, three, BRANCH])
    assert [len(r) for r in rows] == [2, 3, 2]
    assert sorted(len(b["prompt"]) for b in scoring(captured)) == [1, 2]
