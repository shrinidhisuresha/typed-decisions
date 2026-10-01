import math
import pytest
from typed_decisions.scoring import score_noul, score_score


def test_noul_is_the_probability_of_yes_not_a_thresholded_bool():
    ans = score_noul(yes_logit=1.0, no_logit=0.0)
    assert isinstance(ans.noul, float)
    assert math.isclose(ans.noul, math.e / (math.e + 1.0), rel_tol=1e-12)


def test_noul_is_half_when_yes_and_no_are_tied():
    assert math.isclose(score_noul(2.0, 2.0).noul, 0.5, rel_tol=1e-12)


def test_noul_stays_within_the_unit_interval_for_extreme_logits():
    assert 0.0 <= score_noul(-500.0, 500.0).noul <= 1.0
    assert 0.0 <= score_noul(500.0, -500.0).noul <= 1.0


def test_score_is_the_expectation_over_the_legend_not_an_argmax():
    # Evenly split between 1 and 5 must give 3.0, which is not a legend value at all.
    ans = score_score([0.0, 0.0], legend={"1": 1.0, "5": 5.0})
    assert math.isclose(ans.score, 3.0, rel_tol=1e-12)


def test_score_reports_the_legend_it_used():
    ans = score_score([0.0, 0.0], legend={"bad": 1.0, "good": 5.0})
    assert ans.legend == {"bad": 1.0, "good": 5.0}


def test_score_probabilities_are_keyed_by_legend_label():
    ans = score_score([0.0, 10.0], legend={"bad": 1.0, "good": 5.0})
    assert set(ans.probabilities) == {"bad", "good"}
    assert ans.probabilities["good"] > 0.99


def test_score_is_bounded_by_the_legend_range():
    ans = score_score([3.0, -1.0, 0.5], legend={"low": 1.0, "mid": 3.0, "high": 5.0})
    assert 1.0 <= ans.score <= 5.0


def test_score_rejects_a_single_band_legend():
    with pytest.raises(ValueError, match="legend"):
        score_score([1.0], legend={"only": 1.0})
