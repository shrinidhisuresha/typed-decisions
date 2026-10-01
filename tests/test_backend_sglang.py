import json
import httpx
import pytest

from typed_decisions.backends import Branch
from typed_decisions.backend_impls.sglang import SGLangBackend
from tests.doubles import FakeTokenizer

PREFIX = "<state>\nticket: charged twice\n</state>\n"
BRANCH = Branch(text="\nPick one.\nA) billing\nB) sales\nAnswer:", token_ids=[1000, 1001],
                rendered=[" A", " B"])
OTHER = Branch(text="\nUrgent?\nA) yes\nB) no\nAnswer:", token_ids=[1000, 1001],
               rendered=[" A", " B"])


def output(ids_logprobs, prompt_tokens=10, cached_tokens=0):
    return {"text": "A", "meta_info": {
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "output_token_ids_logprobs": [[[lp, tid, None] for tid, lp in ids_logprobs.items()]],
    }}


def make_backend(captured, logprobs={1000: -0.5, 1001: -1.5}, cached_tokens=0, **kwargs):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        if isinstance(body["text"], str):
            return httpx.Response(200, json=output({}, prompt_tokens=8))
        return httpx.Response(200, json=[output(logprobs, cached_tokens=cached_tokens)
                                         for _ in body["text"]])
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://sglang.test")
    return SGLangBackend(tokenizer=FakeTokenizer(), client=client, **kwargs)


def test_uses_the_native_generate_endpoint_with_raw_text_not_chat_messages():
    captured = []
    make_backend(captured).score(PREFIX, [BRANCH])
    assert all("text" in body and "messages" not in body for body in captured)


def test_warms_the_bare_state_first_then_scores_every_branch_in_one_batch():
    captured = []
    make_backend(captured).score(PREFIX, [BRANCH, OTHER])
    assert len(captured) == 2
    assert captured[0]["text"] == PREFIX
    assert captured[1]["text"] == [PREFIX + BRANCH.text, PREFIX + OTHER.text]


def test_every_branch_starts_with_the_state():
    captured = []
    make_backend(captured).score(PREFIX, [BRANCH, OTHER])
    assert all(t.startswith(PREFIX) for t in captured[-1]["text"])


def test_asks_for_the_label_token_ids_by_id():
    captured = []
    make_backend(captured).score(PREFIX, [BRANCH])
    assert captured[-1]["token_ids_logprob"] == [[1000, 1001]]
    assert captured[-1]["return_logprob"] is True


def test_generates_exactly_one_token_position():
    captured = []
    make_backend(captured).score(PREFIX, [BRANCH])
    assert all(body["sampling_params"]["max_new_tokens"] == 1 for body in captured)


def test_returns_logprobs_aligned_to_the_branch_token_order():
    rows = make_backend([], logprobs={1001: -1.5, 1000: -0.5}).score(PREFIX, [BRANCH])
    assert rows == [[-0.5, -1.5]]


def test_raises_when_the_server_omits_a_label_id():
    with pytest.raises(RuntimeError, match="1001"):
        make_backend([], logprobs={1000: -0.5}).score(PREFIX, [BRANCH])


def test_raises_on_a_build_without_token_ids_logprob():
    def handler(request):
        body = json.loads(request.content)
        if isinstance(body["text"], str):
            return httpx.Response(200, json=output({}))
        return httpx.Response(200, json=[{"meta_info": {}} for _ in body["text"]])
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://s.test")
    with pytest.raises(RuntimeError, match="token_ids_logprob"):
        SGLangBackend(tokenizer=FakeTokenizer(), client=client).score(PREFIX, [BRANCH])


def test_records_the_servers_cache_hit_rate():
    backend = make_backend([], cached_tokens=9)
    backend.score(PREFIX, [BRANCH, OTHER])
    assert backend.stats.prompt_tokens == 8 + 10 + 10
    assert backend.stats.cached_tokens == 18
    assert backend.stats.hit_rate == pytest.approx(18 / 28)


def test_no_branches_means_no_requests():
    captured = []
    assert make_backend(captured).score(PREFIX, []) == []
    assert captured == []
