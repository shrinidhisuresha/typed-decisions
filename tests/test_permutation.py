import pytest
from typed_decisions.calibration.permutation import rotations, unpermute, average_distributions


def test_identity_is_the_only_ordering_when_one_permutation_is_requested():
    assert rotations(4, 1) == [(0, 1, 2, 3)]


def test_each_ordering_is_a_permutation_of_the_options():
    for ordering in rotations(4, 4):
        assert sorted(ordering) == [0, 1, 2, 3]


def test_full_rotation_puts_every_option_in_every_position_exactly_once():
    orderings = rotations(4, 4)
    for position in range(4):
        seen = [ordering[position] for ordering in orderings]
        assert sorted(seen) == [0, 1, 2, 3]


def test_requesting_more_permutations_than_options_is_clamped():
    assert len(rotations(3, 99)) == 3


def test_unpermute_sends_each_position_back_to_its_original_option():
    # position 0 holds original option 2, position 1 holds 0, position 2 holds 1
    ordering = (2, 0, 1)
    out = unpermute([0.7, 0.2, 0.1], ordering)
    assert out[2] == pytest.approx(0.7)
    assert out[0] == pytest.approx(0.2)
    assert out[1] == pytest.approx(0.1)


def test_unpermute_of_the_identity_ordering_changes_nothing():
    assert unpermute([0.5, 0.3, 0.2], (0, 1, 2)) == pytest.approx([0.5, 0.3, 0.2])


def test_unpermute_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        unpermute([0.5, 0.5], (0, 1, 2))


def test_averaging_identical_distributions_is_a_no_op():
    d = [0.6, 0.3, 0.1]
    assert average_distributions([d, d, d]) == pytest.approx(d)


def test_averaging_renormalises_to_a_distribution():
    out = average_distributions([[0.8, 0.2], [0.2, 0.8]])
    assert sum(out) == pytest.approx(1.0)
    assert out == pytest.approx([0.5, 0.5])


def test_averaging_rejects_an_empty_list():
    with pytest.raises(ValueError, match="empty"):
        average_distributions([])


def test_rotation_averaging_cancels_a_pure_first_position_bias():
    # Simulate a model that ignores content and always favours whatever sits first.
    n = 3
    biased_by_position = [0.6, 0.2, 0.2]
    collected = []
    for ordering in rotations(n, n):
        collected.append(unpermute(biased_by_position, ordering))
    averaged = average_distributions(collected)
    assert averaged == pytest.approx([1 / 3] * 3)
