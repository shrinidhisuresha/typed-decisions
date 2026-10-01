import pytest
from typed_decisions.types import ChoiceQuestion, ScoreQuestion
from typed_decisions.prompt import render_branch, options_in_order

Q = ChoiceQuestion("Pick one.", ["red", "green", "blue"])


def test_default_ordering_is_the_declared_option_order():
    branch = render_branch(Q)
    assert "A) red" in branch and "B) green" in branch and "C) blue" in branch


def test_an_ordering_relabels_options_by_their_new_position():
    # ordering[i] is the ORIGINAL index shown at position i
    branch = render_branch(Q, ordering=(2, 0, 1))
    assert "A) blue" in branch
    assert "B) red" in branch
    assert "C) green" in branch


def test_options_in_order_returns_the_permuted_option_texts():
    assert options_in_order(Q, (2, 0, 1)) == ["blue", "red", "green"]


def test_options_in_order_defaults_to_the_declared_order():
    assert options_in_order(Q, None) == ["red", "green", "blue"]


def test_permuting_a_score_question_permutes_its_legend_bands():
    q = ScoreQuestion("Rate.", {"low": 1.0, "high": 5.0})
    assert options_in_order(q, (1, 0)) == ["high", "low"]


def test_the_shared_prefix_is_unaffected_by_ordering():
    a = render_branch(Q, ordering=(0, 1, 2))
    b = render_branch(Q, ordering=(2, 1, 0))
    assert a != b
    assert a.endswith("Answer:") and b.endswith("Answer:")


def test_rejects_an_ordering_that_is_not_a_permutation():
    with pytest.raises(ValueError, match="permutation"):
        render_branch(Q, ordering=(0, 0, 1))
