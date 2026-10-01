import math
import pytest
from typed_decisions.scoring import score_choice


def test_choice_picks_highest_logit_option():
    ans = score_choice([2.0, 1.0, 0.0], ["red", "green", "blue"])
    assert ans.choice == "red"


def test_choice_probabilities_are_softmax_over_only_the_options():
    ans = score_choice([2.0, 1.0, 0.0], ["red", "green", "blue"])
    assert math.isclose(sum(ans.probabilities.values()), 1.0, abs_tol=1e-9)
    assert math.isclose(ans.probabilities["red"] / ans.probabilities["green"], math.e, rel_tol=1e-9)


def test_choice_probabilities_keyed_by_option_label_not_token():
    ans = score_choice([0.0, 5.0], ["approve", "deny"])
    assert set(ans.probabilities) == {"approve", "deny"}
    assert ans.choice == "deny"
