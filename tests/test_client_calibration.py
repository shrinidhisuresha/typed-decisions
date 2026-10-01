import pytest
from typed_decisions.client import Decider, Calibration
from typed_decisions.types import ChoiceQuestion
from tests.doubles import RecordingBackend, PositionBiasedBackend, PriorOnlyBackend

STATE = {"ticket": "charged twice"}
Q = {"intent": ChoiceQuestion("Classify the intent.", ["billing", "technical", "sales"])}
LOGITS = {"Classify the intent.": [5.0, 1.0, 0.0]}


def test_default_calibration_is_a_no_op():
    plain = Decider(RecordingBackend(LOGITS)).ask(STATE, Q)
    explicit = Decider(RecordingBackend(LOGITS), calibration=Calibration()).ask(STATE, Q)
    assert plain.answers["intent"].probabilities == pytest.approx(
        explicit.answers["intent"].probabilities
    )


def test_permutations_produce_one_branch_per_ordering():
    backend = RecordingBackend(LOGITS)
    Decider(backend, calibration=Calibration(permutations=3)).ask(STATE, Q)
    _prefix, branches = backend.calls[0]
    assert len(branches) == 3


def test_permutations_still_share_a_single_prefix():
    backend = RecordingBackend(LOGITS)
    Decider(backend, calibration=Calibration(permutations=3)).ask(STATE, Q)
    assert len(backend.calls) == 1, "permutations must fork, not re-prefill"


def test_a_pure_position_bias_is_cancelled_end_to_end():
    backend = PositionBiasedBackend(bias=3.0)
    result = Decider(backend, calibration=Calibration(permutations=3)).ask(STATE, Q)
    probs = result.answers["intent"].probabilities
    assert list(probs.values()) == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-9)


def test_without_permutation_the_same_bias_survives():
    # Guards against the test above passing for the wrong reason.
    backend = PositionBiasedBackend(bias=3.0)
    probs = Decider(backend).ask(STATE, Q).answers["intent"].probabilities
    assert probs["billing"] > 0.8


def test_contextual_calibration_adds_a_null_state_pass():
    backend = RecordingBackend(LOGITS)
    Decider(backend, calibration=Calibration(contextual=True)).ask(STATE, Q)
    assert len(backend.calls) == 2
    prefixes = [call[0] for call in backend.calls]
    assert any("charged twice" in p for p in prefixes)
    assert any("charged twice" not in p for p in prefixes)


def test_an_answer_that_ignores_the_state_is_reduced_to_uniform():
    backend = PriorOnlyBackend([4.0, 1.0, 0.0])
    result = Decider(backend, calibration=Calibration(contextual=True)).ask(STATE, Q)
    probs = result.answers["intent"].probabilities
    assert list(probs.values()) == pytest.approx([1 / 3, 1 / 3, 1 / 3], abs=1e-9)


def test_without_contextual_the_same_uninformative_answer_looks_confident():
    backend = PriorOnlyBackend([4.0, 1.0, 0.0])
    probs = Decider(backend).ask(STATE, Q).answers["intent"].probabilities
    assert probs["billing"] > 0.9


def test_a_fitted_temperature_above_one_lowers_confidence():
    hot = Decider(RecordingBackend(LOGITS), calibration=Calibration(temperature=5.0))
    cold = Decider(RecordingBackend(LOGITS), calibration=Calibration(temperature=1.0))
    assert hot.ask(STATE, Q).answers["intent"].confidence < cold.ask(STATE, Q).answers["intent"].confidence


def test_temperature_never_changes_which_option_wins():
    hot = Decider(RecordingBackend(LOGITS), calibration=Calibration(temperature=9.0))
    assert hot.ask(STATE, Q).answers["intent"].choice == "billing"


def test_calibration_rejects_a_non_positive_permutation_count():
    with pytest.raises(ValueError, match="permutations"):
        Calibration(permutations=0)
