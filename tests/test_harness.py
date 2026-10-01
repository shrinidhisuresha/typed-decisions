import math
import pytest
from typed_decisions.calibration.harness import Prediction, report, fit_temperature_from_predictions


def overconfident(n=100):
    """Says 0.95 every time, right only 60% of the time."""
    right = [Prediction({"a": 0.95, "b": 0.05}, "a") for _ in range(int(n * 0.6))]
    wrong = [Prediction({"a": 0.95, "b": 0.05}, "b") for _ in range(n - int(n * 0.6))]
    return right + wrong


def well_calibrated(n=100):
    right = [Prediction({"a": 0.8, "b": 0.2}, "a") for _ in range(80)]
    wrong = [Prediction({"a": 0.8, "b": 0.2}, "b") for _ in range(20)]
    return right + wrong


def test_report_counts_the_sample_and_accuracy():
    r = report(well_calibrated())
    assert r.count == 100
    assert r.accuracy == pytest.approx(0.8)


def test_well_calibrated_predictions_report_near_zero_ece():
    assert report(well_calibrated()).ece == pytest.approx(0.0, abs=1e-9)


def test_overconfident_predictions_report_a_large_ece():
    assert report(overconfident()).ece == pytest.approx(0.35, abs=1e-9)


def test_report_includes_reliability_bins_covering_the_sample():
    r = report(well_calibrated())
    assert sum(b.count for b in r.reliability) == 100


def test_report_includes_a_selective_accuracy_curve():
    r = report(well_calibrated())
    assert r.selective[0].coverage == pytest.approx(1.0)
    assert all(p.coverage <= 1.0 for p in r.selective)


def test_report_includes_brier_score():
    assert report(well_calibrated()).brier > 0.0
    assert report([Prediction({"a": 1.0, "b": 0.0}, "a")]).brier == pytest.approx(0.0)


def test_report_rejects_an_empty_sample():
    with pytest.raises(ValueError, match="empty"):
        report([])


def test_fitting_on_overconfident_predictions_returns_a_temperature_above_one():
    assert fit_temperature_from_predictions(overconfident()) > 1.0


def test_fitting_measurably_reduces_ece():
    # The entire point of Phase 1. If this does not hold, calibration is not working.
    preds = overconfident()
    before = report(preds).ece
    t = fit_temperature_from_predictions(preds)
    after = report(preds, temperature=t).ece
    assert after < before
    assert after < 0.05


def test_applying_a_temperature_does_not_change_accuracy():
    preds = overconfident()
    t = fit_temperature_from_predictions(preds)
    assert report(preds, temperature=t).accuracy == pytest.approx(report(preds).accuracy)


def test_report_can_measure_the_margin_confidence_a_router_actually_gates_on():
    r = report(well_calibrated(), confidence="margin")
    # margin for 0.8/0.2 is 0.6, accuracy is 0.8 -> a real 0.2 gap
    assert r.ece == pytest.approx(0.2, abs=1e-9)
