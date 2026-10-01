import pytest
from typed_decisions.calibration.temperature import sharpen


def test_temperature_of_one_leaves_the_distribution_untouched():
    p = [0.6, 0.3, 0.1]
    assert sharpen(p, 1.0) == pytest.approx(p)


def test_a_high_temperature_flattens_an_overconfident_distribution():
    out = sharpen([0.9, 0.1], 5.0)
    assert out[0] < 0.9
    assert out[0] > out[1]


def test_a_low_temperature_sharpens_a_hedged_distribution():
    out = sharpen([0.6, 0.4], 0.2)
    assert out[0] > 0.6


def test_output_remains_a_distribution():
    out = sharpen([0.5, 0.3, 0.2], 2.5)
    assert sum(out) == pytest.approx(1.0)


def test_ordering_of_options_is_never_changed_by_temperature():
    out = sharpen([0.5, 0.3, 0.2], 7.0)
    assert out[0] > out[1] > out[2]


def test_a_zero_is_floored_the_same_way_the_fit_floors_it():
    # It used to stay exactly 0 while the fit treated it as 1e-12 -- the inconsistency the
    # head-to-head audit found. Now both sides use the same floor.
    from typed_decisions.calibration.temperature import DEFAULT_FLOOR
    out = sharpen([0.7, 0.3, 0.0], 2.0)
    assert 0.0 < out[2] < 10 * DEFAULT_FLOOR ** 0.5
    assert sum(out) == pytest.approx(1.0)


def test_floor_zero_keeps_zeros_exactly_zero():
    assert sharpen([0.7, 0.3, 0.0], 2.0, floor=0.0)[2] == 0.0


def test_rejects_a_non_positive_temperature():
    with pytest.raises(ValueError, match="positive"):
        sharpen([0.5, 0.5], 0.0)
