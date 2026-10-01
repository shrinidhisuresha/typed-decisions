import math
import pytest
from typed_decisions.calibration.metrics import (
    expected_calibration_error,
    reliability_bins,
    brier_score,
    selective_accuracy,
)


def test_perfectly_calibrated_predictions_have_zero_ece():
    # 10 predictions at 0.8 confidence, exactly 8 correct.
    conf = [0.8] * 10
    correct = [True] * 8 + [False] * 2
    assert expected_calibration_error(conf, correct, bins=10) == pytest.approx(0.0, abs=1e-12)


def test_always_certain_and_always_wrong_has_maximal_ece():
    assert expected_calibration_error([1.0] * 5, [False] * 5) == pytest.approx(1.0)


def test_overconfidence_is_measured_as_the_confidence_accuracy_gap():
    # 0.9 confident, only 50% right -> gap of 0.4
    conf = [0.9] * 10
    correct = [True] * 5 + [False] * 5
    assert expected_calibration_error(conf, correct) == pytest.approx(0.4, abs=1e-12)


def test_ece_weights_bins_by_population():
    # 90 predictions perfectly calibrated, 10 badly wrong -> gap contributes 10%
    conf = [0.5] * 90 + [1.0] * 10
    correct = [True] * 45 + [False] * 45 + [False] * 10
    assert expected_calibration_error(conf, correct, bins=10) == pytest.approx(0.1, abs=1e-12)


def test_ece_rejects_mismatched_input_lengths():
    with pytest.raises(ValueError, match="same length"):
        expected_calibration_error([0.5, 0.5], [True])


def test_ece_rejects_an_empty_sample():
    with pytest.raises(ValueError, match="empty"):
        expected_calibration_error([], [])


def test_reliability_bins_report_count_confidence_and_accuracy():
    rows = reliability_bins([0.05, 0.95, 0.96], [False, True, True], bins=10)
    populated = [r for r in rows if r.count > 0]
    assert len(populated) == 2
    assert populated[0].count == 1
    assert populated[1].count == 2
    assert populated[1].accuracy == pytest.approx(1.0)


def test_reliability_bins_cover_the_unit_interval_without_gaps():
    rows = reliability_bins([0.5], [True], bins=4)
    assert [r.lo for r in rows] == pytest.approx([0.0, 0.25, 0.5, 0.75])
    assert [r.hi for r in rows] == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_confidence_of_exactly_one_lands_in_the_top_bin():
    rows = reliability_bins([1.0], [True], bins=10)
    assert rows[-1].count == 1


def test_brier_score_is_zero_for_a_perfect_confident_prediction():
    assert brier_score([{"a": 1.0, "b": 0.0}], ["a"]) == pytest.approx(0.0)


def test_brier_score_penalises_a_confident_wrong_prediction_most():
    confident_wrong = brier_score([{"a": 1.0, "b": 0.0}], ["b"])
    hedged = brier_score([{"a": 0.5, "b": 0.5}], ["b"])
    assert confident_wrong > hedged
    assert confident_wrong == pytest.approx(2.0)


def test_brier_score_rejects_an_unknown_label():
    with pytest.raises(ValueError, match="not among"):
        brier_score([{"a": 1.0, "b": 0.0}], ["z"])


def test_selective_accuracy_trades_coverage_for_accuracy():
    conf = [0.9, 0.8, 0.4, 0.3]
    correct = [True, True, False, False]
    rows = selective_accuracy(conf, correct, thresholds=[0.0, 0.5])
    assert rows[0].coverage == pytest.approx(1.0)
    assert rows[0].accuracy == pytest.approx(0.5)
    assert rows[1].coverage == pytest.approx(0.5)
    assert rows[1].accuracy == pytest.approx(1.0)


def test_selective_accuracy_reports_undefined_accuracy_when_nothing_clears_the_bar():
    rows = selective_accuracy([0.1], [True], thresholds=[0.9])
    assert rows[0].coverage == pytest.approx(0.0)
    assert rows[0].accuracy is None
