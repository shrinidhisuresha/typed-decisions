import numpy as np
import pytest

from typed_decisions.finetune.rescore import (
    Policy,
    calibration_line,
    cross_fit,
    ece_equal_mass,
    ece_floor,
    fit_t,
    pooled_ci,
    rps_rows,
    scale,
    verdict,
)


def calibrated(n=4000, k=4, seed=0):
    """Predictions, and labels drawn FROM those predictions: calibrated by construction."""
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(k) * 0.7, size=n)
    y = np.array([rng.choice(k, p=row) for row in p])
    return p, y


def test_temperature_and_calibration_line_recover_a_calibrated_model():
    p, y = calibrated(n=20000)                 # the slope estimate is noisy below ~10k rows
    assert fit_t(p, y) == pytest.approx(1.0, abs=0.1)
    intercept, slope = calibration_line(p, y)
    assert abs(intercept) < 0.15 and slope == pytest.approx(1.0, abs=0.12)


def test_overconfidence_shows_as_t_above_one_and_slope_below_one():
    p, y = calibrated()
    sharp = scale(p, 0.4)                      # the same model, made overconfident
    assert fit_t(sharp, y) == pytest.approx(2.5, rel=0.15)
    assert calibration_line(sharp, y)[1] < 0.6
    fixed, ts = cross_fit(sharp, y)
    assert all(t == pytest.approx(2.5, rel=0.2) for t in ts)
    assert ece_equal_mass(fixed, y) < ece_equal_mass(sharp, y)


def test_ece_floor_is_what_a_calibrated_model_scores_at_small_n():
    p, y = calibrated(n=75)
    floor = ece_floor(p)
    assert 0.03 < floor < 0.2                  # small-n binning noise, not miscalibration
    big_p, _ = calibrated(n=3000)
    assert ece_floor(big_p) < floor


def test_rps_charges_ordinal_distance():
    truth = np.array([2])
    near = rps_rows(np.array([[0.0, 1.0, 0.0]]), truth)
    far = rps_rows(np.array([[1.0, 0.0, 0.0]]), truth)
    assert near[0] < far[0]


@pytest.mark.parametrize("brier, acc, expected", [
    ((-0.03, 0.01), (-0.01, 0.05), "PASS"),
    ((0.03, 0.08), (-0.01, 0.05), "FAIL"),
    ((-0.01, 0.01), (-0.10, -0.03), "FAIL"),
    ((-0.04, 0.06), (-0.05, 0.05), "INCONCLUSIVE"),
])
def test_verdict_is_three_way(brier, acc, expected):
    assert verdict(brier, acc, Policy()) == expected


def test_two_level_bootstrap_is_wider_when_seeds_disagree():
    rng = np.random.default_rng(0)
    agree = [[rng.normal(0.0, 0.1, 200) for _ in range(3)] for _ in range(3)]
    disagree = [[rng.normal(mu, 0.1, 200) for _ in range(3)] for mu in (-0.05, 0.0, 0.05)]
    _, lo_a, hi_a = pooled_ci(agree[0], Policy(resamples=500), agree)
    _, lo_d, hi_d = pooled_ci(disagree[0], Policy(resamples=500), disagree)
    assert hi_d - lo_d > hi_a - lo_a
