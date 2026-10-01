import math
import pytest
from typed_decisions.scoring import score_choice, confidence_from


def test_default_confidence_is_the_top_two_margin():
    ans = score_choice([0.0, 5.0], ["approve", "deny"])
    p = ans.probabilities
    assert math.isclose(ans.confidence, p["deny"] - p["approve"], rel_tol=1e-12)


def test_uniform_distribution_has_zero_margin_confidence():
    ans = score_choice([1.0, 1.0, 1.0], ["a", "b", "c"])
    assert math.isclose(ans.confidence, 0.0, abs_tol=1e-12)


def test_max_prob_confidence_mode():
    ans = score_choice([0.0, 5.0], ["approve", "deny"], confidence="max_prob")
    assert math.isclose(ans.confidence, max(ans.probabilities.values()), rel_tol=1e-12)


def test_normalized_entropy_confidence_is_one_for_a_certain_answer():
    assert math.isclose(confidence_from([1.0, 0.0, 0.0], mode="entropy"), 1.0, abs_tol=1e-9)


def test_normalized_entropy_confidence_is_zero_for_a_uniform_answer():
    assert math.isclose(confidence_from([1 / 3, 1 / 3, 1 / 3], mode="entropy"), 0.0, abs_tol=1e-9)


def test_entropy_confidence_is_comparable_across_different_option_counts():
    # normalized by log N, so a uniform 2-way and a uniform 10-way both score 0
    two = confidence_from([0.5, 0.5], mode="entropy")
    ten = confidence_from([0.1] * 10, mode="entropy")
    assert math.isclose(two, ten, abs_tol=1e-9)


def test_unknown_confidence_mode_is_rejected():
    with pytest.raises(ValueError, match="confidence"):
        score_choice([1.0, 2.0], ["a", "b"], confidence="vibes")
