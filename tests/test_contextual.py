import math
import pytest
from typed_decisions.calibration.contextual import debias, NULL_STATE


def test_a_uniform_null_prior_leaves_the_distribution_untouched():
    probs = [0.6, 0.3, 0.1]
    out = debias(probs, [1 / 3, 1 / 3, 1 / 3])
    assert out == pytest.approx(probs)


def test_an_answer_that_merely_echoes_the_prior_carries_no_information():
    # If the model says exactly what it says with no state at all, the calibrated
    # answer must be uniform -- the state told us nothing.
    prior = [0.7, 0.2, 0.1]
    assert debias(prior, prior) == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_an_option_the_model_prefers_a_priori_is_suppressed():
    # Model says 0.5/0.5, but with no state it already leaned 0.9 towards A.
    out = debias([0.5, 0.5], [0.9, 0.1])
    assert out[1] > out[0]


def test_output_is_a_probability_distribution():
    out = debias([0.5, 0.3, 0.2], [0.6, 0.3, 0.1])
    assert sum(out) == pytest.approx(1.0)
    assert all(p >= 0.0 for p in out)


def test_relative_odds_are_divided_by_the_prior_odds():
    out = debias([0.5, 0.5], [0.8, 0.2])
    assert out[0] / out[1] == pytest.approx((0.5 / 0.5) / (0.8 / 0.2))


def test_a_zero_prior_does_not_blow_up():
    out = debias([0.5, 0.5], [0.0, 1.0])
    assert sum(out) == pytest.approx(1.0)
    assert all(math.isfinite(p) for p in out)


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        debias([0.5, 0.5], [1.0])


def test_rejects_an_all_zero_prior():
    with pytest.raises(ValueError, match="prior"):
        debias([0.5, 0.5], [0.0, 0.0])


def test_null_state_is_content_free():
    assert NULL_STATE.strip() in {"N/A", "[MASK]", ""}
