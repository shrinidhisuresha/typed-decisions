import pytest

from typed_decisions.gate import GatePolicy, compare, paired_interval

RIGHT, WRONG = [0.9, 0.1], [0.1, 0.9]


def rows(n, student, teacher):
    return [student] * n, [teacher] * n, [0] * n


def test_a_clearly_better_student_with_enough_evidence_is_promoted():
    v = compare(*rows(60, RIGHT, [0.6, 0.4]))
    assert v["promoted"] and v["reasons"] == ["ok"]
    assert v["brier_diff_ci"][1] < 0


def test_too_few_rows_fail_the_floor_even_when_better():
    v = compare(*rows(10, RIGHT, [0.6, 0.4]))
    assert not v["promoted"] and "only 10" in v["reasons"][0]


def test_better_calibrated_but_less_accurate_is_refused():
    # the case the old Brier-only gate let through
    student = [[0.55, 0.45]] * 40 + [[0.45, 0.55]] * 20        # right 40/60, hedged
    teacher = [[0.99, 0.01]] * 50 + [[0.01, 0.99]] * 10        # right 50/60, sharp
    v = compare(student, teacher, [0] * 60)
    assert not v["promoted"]
    assert any("accuracy may drop" in r for r in v["reasons"])
    assert v["teacher_only_right"] == 10 and v["student_only_right"] == 0


def test_a_noisy_tie_on_small_data_is_refused_by_the_interval_not_the_point_estimate():
    student = [RIGHT, WRONG] * 20
    teacher = [WRONG, RIGHT] * 20
    v = compare(student, teacher, [0] * 40)
    assert v["student_accuracy"] == v["teacher_accuracy"]
    assert not v["promoted"]


def test_identical_models_are_promotable_under_non_inferiority():
    v = compare(*rows(60, RIGHT, RIGHT))
    assert v["promoted"], v["reasons"]


def test_interval_is_deterministic_and_brackets_the_mean():
    diffs = [0.1, -0.2, 0.3, 0.0, 0.05] * 10
    policy = GatePolicy()
    lo, hi = paired_interval(diffs, policy)
    assert paired_interval(diffs, policy) == (lo, hi)
    assert lo <= sum(diffs) / len(diffs) <= hi


def test_empty_and_misaligned_inputs():
    assert compare([], [], [])["reasons"] == ["no held-out outcomes"]
    with pytest.raises(ValueError):
        compare([RIGHT], [], [0])


def test_the_policy_is_recorded_with_the_verdict():
    v = compare(*rows(60, RIGHT, RIGHT), GatePolicy(max_accuracy_drop=0.1))
    assert v["policy"]["max_accuracy_drop"] == 0.1
