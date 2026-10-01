import pytest
from typed_decisions.calibration.harness import Prediction, diagnose_temperature


def overconfident(n=100):
    right = [Prediction({"a": 0.95, "b": 0.05}, "a") for _ in range(60)]
    wrong = [Prediction({"a": 0.95, "b": 0.05}, "b") for _ in range(40)]
    return right + wrong


def always_right_and_certain():
    # Nothing to learn: the fit will run to the sharpening bound.
    return [Prediction({"a": 0.6, "b": 0.4}, "a") for _ in range(50)]


def test_a_healthy_fit_is_not_flagged_as_degenerate():
    fit = diagnose_temperature(overconfident())
    assert fit.at_bound is False


def test_a_fit_that_runs_to_the_search_bound_is_flagged():
    # T pinned at a bound means the data wants a more extreme scaling than the search
    # allows -- usually too few examples, or a distribution the fit cannot express.
    fit = diagnose_temperature(always_right_and_certain(), bounds=(0.05, 20.0))
    assert fit.at_bound is True


def test_the_fit_reports_nll_before_and_after():
    fit = diagnose_temperature(overconfident())
    assert fit.nll_after <= fit.nll_before


def test_the_fit_reports_the_temperature_it_found():
    fit = diagnose_temperature(overconfident())
    assert fit.temperature > 1.0


def test_improved_is_false_when_the_fit_cannot_beat_no_scaling():
    already = [Prediction({"a": 0.8, "b": 0.2}, "a") for _ in range(80)]
    already += [Prediction({"a": 0.8, "b": 0.2}, "b") for _ in range(20)]
    fit = diagnose_temperature(already)
    assert fit.temperature == pytest.approx(1.0, abs=0.05)
    assert fit.nll_after == pytest.approx(fit.nll_before, abs=1e-3)


def test_diagnostics_reject_an_empty_sample():
    with pytest.raises(ValueError, match="empty"):
        diagnose_temperature([])


def test_a_perfectly_ranked_sample_is_flagged_separable():
    from typed_decisions import Prediction
    preds = [Prediction({"a": 0.7, "b": 0.3}, "a"), Prediction({"a": 0.2, "b": 0.8}, "b")]
    fit = diagnose_temperature(preds)
    assert fit.separable and not fit.trustworthy
    assert "SEPARABLE" in fit.summary()


def test_a_sample_with_a_mistake_is_not_separable():
    from typed_decisions import Prediction
    preds = [Prediction({"a": 0.7, "b": 0.3}, "a"), Prediction({"a": 0.6, "b": 0.4}, "b")]
    assert not diagnose_temperature(preds).separable
