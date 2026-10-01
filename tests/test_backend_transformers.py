import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from typed_decisions.backends import Branch
from typed_decisions.backend_impls.transformers import TransformersBackend

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
PREFIX = "<state>\nticket: the customer was charged twice this month\n</state>\n"
BRANCHES = [
    Branch("\nPick one.\nA) billing\nB) sales\nAnswer:", [319, 350], [" A", " B"]),
    Branch("\nUrgent?\nA) yes\nB) no\nAnswer:", [319, 350], [" A", " B"]),
]


@pytest.fixture(scope="module")
def backend():
    return TransformersBackend.from_pretrained(TINY, device="cpu")


def test_returns_one_row_per_branch_aligned_to_token_ids(backend):
    rows = backend.score(PREFIX, BRANCHES)
    assert len(rows) == 2
    assert all(len(row) == 2 for row in rows)


def test_branch_is_tokenised_in_context_not_standalone(backend):
    # SentencePiece prepends an artifact token to a standalone fragment, and BPE
    # merges across the seam. Encoding the branch alone therefore scores a
    # different token sequence than the model would really see.
    shared_ids, branch_id_lists = backend.encode_branches(PREFIX, BRANCHES)
    for branch, branch_ids in zip(BRANCHES, branch_id_lists):
        whole = backend.tokenizer.encode(PREFIX + branch.text)
        assert shared_ids + branch_ids == whole


def test_forked_cache_gives_the_same_logits_as_scoring_each_prompt_whole(backend):
    # The optimisation must be invisible in the output. If this drifts, the cache
    # fork is wrong (bad positions, stale mask) and every probability is suspect.
    forked = backend.score(PREFIX, BRANCHES)
    naive = [backend.score_uncached(PREFIX, b) for b in BRANCHES]
    for row_f, row_n in zip(forked, naive):
        for a, b in zip(row_f, row_n):
            assert a == pytest.approx(b, abs=1e-3)


def test_prefix_is_prefilled_once_regardless_of_branch_count(backend):
    seen = []
    original = backend.model.forward

    def spy(*args, **kwargs):
        ids = kwargs.get("input_ids")
        seen.append(ids.shape[-1] if ids is not None else 0)
        return original(*args, **kwargs)

    backend.model.forward = spy
    try:
        backend.score(PREFIX, BRANCHES)
    finally:
        backend.model.forward = original

    shared_ids, _ = backend.encode_branches(PREFIX, BRANCHES)
    assert seen.count(len(shared_ids)) == 1, "state was prefilled more than once"
    assert len(seen) == 1 + len(BRANCHES)


def test_branch_passes_do_not_resend_the_prefix(backend):
    seen = []
    original = backend.model.forward

    def spy(*args, **kwargs):
        ids = kwargs.get("input_ids")
        seen.append(ids.shape[-1] if ids is not None else 0)
        return original(*args, **kwargs)

    backend.model.forward = spy
    try:
        backend.score(PREFIX, BRANCHES)
    finally:
        backend.model.forward = original

    shared_ids, _ = backend.encode_branches(PREFIX, BRANCHES)
    for length in seen[1:]:
        assert length < len(shared_ids)


def test_scoring_twice_gives_identical_results(backend):
    # Catches a cache that is mutated by the first branch and not restored.
    first = backend.score(PREFIX, BRANCHES)
    second = backend.score(PREFIX, BRANCHES)
    flat = lambda rows: [v for row in rows for v in row]
    assert flat(first) == pytest.approx(flat(second), abs=1e-6)


SEQ = Branch("\nWhich team?\nOptions:\n- billing\n- technical support\n- sales\nAnswer:",
             [], [], continuations=(" billing", " technical support", " sales"))


def test_sequence_branch_returns_one_mean_logprob_per_option(backend):
    rows = backend.score(PREFIX, [SEQ])
    assert len(rows[0]) == 3
    assert all(score <= 0.0 for score in rows[0])


def test_forked_sequence_scores_match_a_whole_prompt_forward(backend):
    # Two levels of fork (state -> question -> option); must still be invisible.
    forked = backend.score(PREFIX, [SEQ])[0]
    naive = backend.score_sequence_uncached(PREFIX, SEQ)
    assert forked == pytest.approx(naive, abs=1e-3)


def test_letter_and_sequence_branches_mix_in_one_call(backend):
    rows = backend.score(PREFIX, [BRANCHES[0], SEQ, BRANCHES[1]])
    assert [len(r) for r in rows] == [2, 3, 2]
    assert rows[0] == pytest.approx(backend.score(PREFIX, [BRANCHES[0]])[0], abs=1e-4)


def test_sum_normalisation_is_mean_times_length_and_still_matches_uncached(backend):
    import dataclasses
    summed = dataclasses.replace(SEQ, normalize="sum")
    forked = backend.score(PREFIX, [summed])[0]
    assert forked == pytest.approx(backend.score_sequence_uncached(PREFIX, summed), abs=1e-3)
    means = backend.score(PREFIX, [SEQ])[0]
    lengths = [len(backend.continuation_ids(backend.tokenizer.encode(PREFIX + SEQ.text),
                                            PREFIX + SEQ.text, o)) for o in SEQ.continuations]
    assert forked == pytest.approx([m * n for m, n in zip(means, lengths)], abs=1e-3)
