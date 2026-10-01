import math
import pytest
from typed_decisions.calibration.temperature import fit_temperature, negative_log_likelihood, sharpen


def confident_but_half_wrong():
    # Sharply peaked logits, right only half the time -> overconfident.
    rows = [[5.0, 0.0]] * 20
    labels = [0] * 10 + [1] * 10
    return rows, labels


def hedged_but_always_right():
    # Nearly flat logits, always right -> underconfident.
    rows = [[0.1, 0.0]] * 20
    labels = [0] * 20
    return rows, labels


def test_overconfident_predictions_are_softened_by_a_temperature_above_one():
    rows, labels = confident_but_half_wrong()
    assert fit_temperature(rows, labels) > 1.0


def test_underconfident_predictions_are_sharpened_by_a_temperature_below_one():
    rows, labels = hedged_but_always_right()
    assert fit_temperature(rows, labels) < 1.0


def test_fitting_never_increases_negative_log_likelihood():
    rows, labels = confident_but_half_wrong()
    fitted = fit_temperature(rows, labels)
    assert negative_log_likelihood(rows, labels, fitted) <= negative_log_likelihood(rows, labels, 1.0)


def test_already_calibrated_data_keeps_a_temperature_near_one():
    # p(correct) = softmax([log 4, 0]) = 0.8 for the top class; make it right 80% of the time.
    rows = [[math.log(4.0), 0.0]] * 100
    labels = [0] * 80 + [1] * 20
    assert fit_temperature(rows, labels) == pytest.approx(1.0, abs=0.05)


def test_temperature_is_always_positive():
    rows, labels = hedged_but_always_right()
    assert fit_temperature(rows, labels) > 0.0


def test_nll_of_a_perfect_confident_prediction_is_near_zero():
    assert negative_log_likelihood([[100.0, 0.0]], [0], 1.0) == pytest.approx(0.0, abs=1e-9)


def test_nll_penalises_a_confident_wrong_prediction_heavily():
    assert negative_log_likelihood([[10.0, 0.0]], [1], 1.0) > 9.0


def test_fitting_rejects_an_empty_sample():
    with pytest.raises(ValueError, match="empty"):
        fit_temperature([], [])


def test_fitting_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        fit_temperature([[1.0, 0.0]], [0, 1])


def test_fitting_rejects_a_label_outside_the_option_range():
    with pytest.raises(ValueError, match="out of range"):
        fit_temperature([[1.0, 0.0]], [5])


def test_the_fit_and_the_application_treat_zeros_identically():
    """Regression: the fit used to floor zeros at 1e-12 (about 0.25 relative mass at T=20)
    while sharpen() kept them at 0, so the fitted T was optimised for a different
    distribution than the one produced."""
    import math
    from typed_decisions.calibration.temperature import floored, negative_log_likelihood
    probs = [0.7, 0.3, 0.0]
    for t in (0.5, 3.0, 20.0):
        applied = sharpen(probs, t, floor=0.005)
        rows = [[math.log(q) for q in floored(probs, 0.005)]]
        # NLL of the true class under what the fit evaluates == -log of what is applied
        nll = negative_log_likelihood(rows, [2], t)
        assert nll == pytest.approx(-math.log(applied[2]), rel=1e-9)


def test_a_zero_on_the_true_label_no_longer_drives_the_fit_to_its_bound():
    from typed_decisions import Prediction, diagnose_temperature
    # confident and mostly right, but a zero on the label for some rows (rounded outputs)
    preds = ([Prediction({"a": 0.99, "b": 0.01, "c": 0.0}, "a")] * 40 +
             [Prediction({"a": 0.99, "b": 0.01, "c": 0.0}, "c")] * 4)
    fit = diagnose_temperature(preds, floor=0.005)
    assert not fit.at_bound and fit.temperature > 1.0      # softens, within bounds


def test_temperature_one_is_the_identity():
    assert sharpen([0.5, 0.5, 0.0], 1.0) == [0.5, 0.5, 0.0]
