import dataclasses

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from typed_decisions.backends import Branch
from typed_decisions.backend_impls.mlx import MLXBackend

TOKENIZER = "hf-internal-testing/tiny-random-LlamaForCausalLM"
PREFIX = "<state>\nticket: the customer was charged twice this month\n</state>\n"


@pytest.fixture(scope="module")
def backend():
    """A tiny random Llama built in MLX: exercises the cache fork, not model quality."""
    from mlx_lm.models import llama
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    args = llama.ModelArgs(model_type="llama", hidden_size=64, num_hidden_layers=2,
                           intermediate_size=128, num_attention_heads=4, rms_norm_eps=1e-5,
                           vocab_size=len(tok), num_key_value_heads=2)
    mx.random.seed(0)
    model = llama.Model(args)
    mx.eval(model.parameters())
    return MLXBackend(model, tok)


def label_ids(backend, letters):
    return [backend.tokenizer.encode(f" {c}")[0] for c in letters]


@pytest.fixture
def branches(backend):
    ab = label_ids(backend, "AB")
    return [Branch("\nPick one.\nA) billing\nB) sales\nAnswer:", ab, [" A", " B"]),
            Branch("\nUrgent?\nA) yes\nB) no\nAnswer:", ab, [" A", " B"])]


SEQ = Branch("\nWhich team?\nOptions:\n- billing\n- technical support\n- sales\nAnswer:",
             [], [], continuations=(" billing", " technical support", " sales"))


def test_forked_cache_gives_the_same_logits_as_scoring_each_prompt_whole(backend, branches):
    forked = backend.score(PREFIX, branches)
    for row, branch in zip(forked, branches):
        assert row == pytest.approx(backend.score_uncached(PREFIX, branch), abs=1e-3)


def test_forked_option_scores_match_a_whole_prompt_forward(backend):
    for norm in ("mean", "sum"):
        b = dataclasses.replace(SEQ, normalize=norm)
        # relative: a "sum" score adds several log-probs, and cached vs whole-prompt
        # attention accumulate in a different order (seen: 1e-3 on -22, i.e. 5e-5 relative)
        assert backend.score(PREFIX, [b])[0] == pytest.approx(
            backend.score_sequence_uncached(PREFIX, b), rel=1e-4, abs=1e-3)


def test_forking_does_not_disturb_the_shared_prefix(backend, branches):
    # scoring the same branch twice in one call must give the same answer: the second
    # fork must not see the first branch's tokens
    again = backend.score(PREFIX, [branches[0], branches[0]])
    assert again[0] == pytest.approx(again[1], abs=1e-6)


def test_letter_and_option_branches_mix(backend, branches):
    rows = backend.score(PREFIX, [branches[0], SEQ])
    assert [len(r) for r in rows] == [2, 3]


def test_stats_count_the_shared_prefix_as_cached(backend, branches):
    backend.stats.prompt_tokens = backend.stats.cached_tokens = 0
    backend.score(PREFIX, branches)
    assert backend.stats.cached_tokens > 0 and backend.stats.hit_rate < 1.0


def test_works_under_the_client(backend):
    from typed_decisions import ChoiceQuestion, Decider
    answer = Decider(backend).ask({"ticket": "x"}, {
        "q": ChoiceQuestion("Which?", ["billing", "sales", "technical"])}).answers["q"]
    assert sum(answer.probabilities.values()) == pytest.approx(1.0)
