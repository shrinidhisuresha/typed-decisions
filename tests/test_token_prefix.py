import pytest
from typed_decisions.backend_impls.transformers import shared_prefix_length, PrefixError


def test_exact_token_prefix_shares_the_whole_state():
    assert shared_prefix_length([1, 2, 3], [[1, 2, 3, 9], [1, 2, 3, 8, 7]]) == 3


def test_a_seam_merge_costs_exactly_one_token_of_sharing():
    # Real case: state ends "\n", branch starts "\n", BPE merges them into one token,
    # so the final state token is replaced in context. We keep sharing the rest.
    assert shared_prefix_length([1, 2, 3], [[1, 2, 99, 9], [1, 2, 99, 8]]) == 2


def test_shared_length_is_capped_at_the_state_even_for_a_single_branch():
    # Otherwise one question would cache its own question text as "state".
    assert shared_prefix_length([1, 2, 3], [[1, 2, 3, 9, 8]]) == 3


def test_divergence_well_before_the_seam_is_a_broken_template():
    with pytest.raises(PrefixError, match="token-level prefix"):
        shared_prefix_length([1, 2, 3, 4, 5], [[1, 9, 9, 9, 9]])


def test_tolerance_is_configurable_for_tokenizers_that_merge_more():
    assert shared_prefix_length([1, 2, 3, 4], [[1, 2, 88, 77]], tolerance=2) == 2
    with pytest.raises(PrefixError):
        shared_prefix_length([1, 2, 3, 4], [[1, 2, 88, 77]], tolerance=1)


def test_branches_that_disagree_with_each_other_share_only_their_common_part():
    assert shared_prefix_length([1, 2, 3], [[1, 2, 3, 9], [1, 2, 55, 8]]) == 2
