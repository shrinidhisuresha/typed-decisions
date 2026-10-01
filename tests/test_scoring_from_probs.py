import pytest
from typed_decisions.scoring import choice_from_probs, noul_from_probs, score_from_probs


def test_choice_from_probs_uses_the_distribution_as_given():
    ans = choice_from_probs([0.2, 0.7, 0.1], ["a", "b", "c"])
    assert ans.choice == "b"
    assert ans.probabilities == pytest.approx({"a": 0.2, "b": 0.7, "c": 0.1})


def test_choice_from_probs_derives_confidence_from_that_distribution():
    ans = choice_from_probs([0.2, 0.7, 0.1], ["a", "b", "c"])
    assert ans.confidence == pytest.approx(0.5)


def test_choice_from_probs_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        choice_from_probs([0.5, 0.5], ["a", "b", "c"])


def test_noul_from_probs_passes_the_probability_through():
    assert noul_from_probs(0.73).noul == pytest.approx(0.73)


def test_score_from_probs_is_the_expectation_over_the_legend():
    ans = score_from_probs([0.5, 0.5], {"low": 1.0, "high": 5.0})
    assert ans.score == pytest.approx(3.0)


def test_score_from_probs_keeps_the_legend_and_probabilities():
    ans = score_from_probs([0.25, 0.75], {"low": 1.0, "high": 5.0})
    assert ans.legend == {"low": 1.0, "high": 5.0}
    assert ans.probabilities == pytest.approx({"low": 0.25, "high": 0.75})
