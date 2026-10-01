import pytest

from typed_decisions.client import MAX_CHOICE_OPTIONS, Calibration, Decider
from typed_decisions.types import ChoiceQuestion, NoulQuestion
from tests.doubles import PositionBiasedBackend, RecordingBackend

STATE = {"ticket": "x"}
MANY = [f"team {i}" for i in range(40)]


class ByText:
    """Scores sequence branches by option text, so the favourite is identifiable."""

    def __init__(self, favourite):
        from tests.doubles import FakeTokenizer
        self.tokenizer = FakeTokenizer()
        self.favourite = favourite
        self.calls = []

    def score(self, prefix, branches):
        self.calls.append(list(branches))
        return [[0.0 if c.strip() == self.favourite else -3.0 for c in b.continuations]
                if b.continuations else [0.0] * b.n_options for b in branches]


def test_more_than_26_options_are_scored_by_option_text():
    backend = ByText("team 33")
    answer = Decider(backend).ask(STATE, {"q": ChoiceQuestion("Which?", MANY)}).answers["q"]
    branch = backend.calls[0][0]
    assert branch.continuations == tuple(f" {o}" for o in MANY)
    assert "A)" not in branch.text and "- team 39" in branch.text
    assert answer.choice == "team 33"


def test_26_or_fewer_options_keep_letter_scoring():
    backend = ByText("x")
    Decider(backend).ask(STATE, {"q": ChoiceQuestion("Which?", MANY[:26])})
    assert backend.calls[0][0].continuations == ()


def test_threshold_zero_forces_sequence_scoring_and_other_types_are_untouched():
    backend = ByText("team 1")
    result = Decider(backend, sequence_threshold=0).ask(
        STATE, {"q": ChoiceQuestion("Which?", MANY[:3]), "u": NoulQuestion("Urgent?")})
    kinds = [bool(b.continuations) for b in backend.calls[0]]
    assert kinds == [True, False]
    assert result.answers["q"].choice == "team 1"


def test_permutations_rotate_the_listing_and_undo_it():
    backend = PositionBiasedBackend(bias=5.0)
    answer = Decider(backend, calibration=Calibration(permutations=len(MANY))).ask(
        STATE, {"q": ChoiceQuestion("Which?", MANY)}).answers["q"]
    firsts = {b.continuations[0] for b in backend.calls[0][1]}
    assert len(firsts) == len(MANY)                        # every option led once
    assert max(answer.probabilities.values()) == pytest.approx(1 / len(MANY))


def test_contextual_calibration_applies_to_sequence_scores():
    backend = ByText("team 5")      # same preference with or without a state
    answer = Decider(backend, calibration=Calibration(contextual=True)).ask(
        STATE, {"q": ChoiceQuestion("Which?", MANY)}).answers["q"]
    assert max(answer.probabilities.values()) == pytest.approx(1 / len(MANY))


def test_over_the_ceiling_is_refused():
    options = [f"o{i}" for i in range(MAX_CHOICE_OPTIONS + 1)]
    with pytest.raises(ValueError, match="retrieval"):
        Decider(RecordingBackend({})).ask(STATE, {"q": ChoiceQuestion("Which?", options)})


def lp_for(text):
    return -0.1 if text.endswith(" billing") else -2.0


def sglang_backend(captured, bad_ids=False):
    import json
    import httpx
    from typed_decisions.backend_impls.sglang import SGLangBackend
    from tests.doubles import FakeTokenizer

    def handler(request):
        body = json.loads(request.content)
        captured.append(body)
        if isinstance(body["text"], str):
            return httpx.Response(200, json={"meta_info": {"prompt_tokens": 5, "cached_tokens": 0}})
        if "token_ids_logprob" in body:      # letter branches
            return httpx.Response(200, json=[{"meta_info": {"output_token_ids_logprobs":
                [[[-0.5, 1000, None], [-1.5, 1001, None]]]}} for _ in body["text"]])
        out = []
        for text, start in zip(body["text"], body["logprob_start_len"]):
            n = len(text) - start
            out.append({"meta_info": {"prompt_tokens": len(text), "cached_tokens": start,
                                      "input_token_logprobs": [[lp_for(text), 8 if bad_ids else 7, None]] * n}})
        return httpx.Response(200, json=out)
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://s.test")
    return SGLangBackend(tokenizer=FakeTokenizer(), client=client)


def vllm_backend(captured, bad_ids=False):
    import json
    import httpx
    from typed_decisions.backend_impls.vllm import VLLMBackend
    from tests.doubles import FakeTokenizer

    def handler(request):
        body = json.loads(request.content)
        captured.append(body)
        prompts = body["prompt"] if isinstance(body["prompt"], list) else [body["prompt"]]
        choices = []
        for i, text in enumerate(prompts):
            if body.get("echo"):
                tid = "token_id:8" if bad_ids else "token_id:7"
                logprobs = {"tokens": [tid] * len(text) + ["token_id:9"],
                            "token_logprobs": [None] + [lp_for(text)] * (len(text) - 1) + [-0.5]}
            else:
                logprobs = {"top_logprobs": [{"token_id:1000": -0.5, "token_id:1001": -1.5}]}
            choices.append({"index": i, "logprobs": logprobs})
        return httpx.Response(200, json={"choices": choices, "usage": {"prompt_tokens": 1}})
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://v.test")
    # FakeTokenizer cannot encode " A" in context, so letters take the cache_letters path
    return VLLMBackend(model="m", tokenizer=FakeTokenizer(), client=client, cache_letters=True)


def seq_branch():
    from typed_decisions.backends import Branch
    return Branch("\nWhich?\nAnswer:", [], [], continuations=(" billing", " sales"))


def letter_branch():
    from typed_decisions.backends import Branch
    return Branch("\nUrgent?\nA) yes\nB) no\nAnswer:", [1000, 1001], [" A", " B"])


@pytest.mark.parametrize("make", [sglang_backend, vllm_backend])
def test_http_backends_score_options_as_mean_logprobs_in_one_request(make):
    captured = []
    rows = make(captured).score("<state>x</state>", [seq_branch()])
    assert rows == [[pytest.approx(-0.1), pytest.approx(-2.0)]]
    scoring = [b for b in captured if isinstance(b.get("text", b.get("prompt")), list)]
    assert len(scoring) == 1
    texts = scoring[0].get("text") or scoring[0].get("prompt")
    assert texts == ["<state>x</state>\nWhich?\nAnswer: billing",
                     "<state>x</state>\nWhich?\nAnswer: sales"]


@pytest.mark.parametrize("make", [sglang_backend, vllm_backend])
def test_http_backends_mix_letter_and_sequence_branches(make):
    rows = make([]).score("<state>x</state>", [letter_branch(), seq_branch()])
    assert rows[0] == [-0.5, -1.5]
    assert rows[1] == [pytest.approx(-0.1), pytest.approx(-2.0)]


@pytest.mark.parametrize("make", [sglang_backend, vllm_backend])
def test_http_backends_refuse_a_client_server_tokenizer_mismatch(make):
    with pytest.raises(RuntimeError, match="tokenizer mismatch"):
        make([], bad_ids=True).score("<state>x</state>", [seq_branch()])


def test_vllm_asks_for_echoed_prompt_logprobs_by_token_id():
    captured = []
    vllm_backend(captured).score("<state>x</state>", [seq_branch()])
    body = [b for b in captured if b.get("echo")][0]
    assert body["logprobs"] >= 0 and body["return_tokens_as_token_ids"] is True


def test_sglang_starts_logprobs_just_before_each_option():
    captured = []
    sglang_backend(captured).score("<state>x</state>", [seq_branch()])
    body = [b for b in captured if isinstance(b.get("logprob_start_len"), list)][0]
    context_len = len("<state>x</state>\nWhich?\nAnswer:")
    assert body["logprob_start_len"] == [context_len - 1, context_len - 1]


def test_sequence_norm_is_passed_to_every_sequence_branch_and_validated():
    backend = ByText("team 3")
    Decider(backend, sequence_norm="sum").ask(STATE, {"q": ChoiceQuestion("Which?", MANY)})
    assert backend.calls[0][0].normalize == "sum"
    with pytest.raises(ValueError, match="sequence_norm"):
        Decider(backend, sequence_norm="median")


@pytest.mark.parametrize("make", [sglang_backend, vllm_backend])
def test_http_backends_honour_sum_normalisation(make):
    import dataclasses
    rows = make([]).score("<state>x</state>", [dataclasses.replace(seq_branch(), normalize="sum")])
    # FakeTokenizer: " billing" is 8 tokens of -0.1, " sales" 6 tokens of -2.0
    assert rows == [[pytest.approx(-0.8), pytest.approx(-12.0)]]
