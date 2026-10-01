import pytest
from typed_decisions.client import Decider, Calibration
from typed_decisions.types import ChoiceQuestion
from tests.doubles import RecordingBackend

Q = {"intent": ChoiceQuestion("Classify the intent.", ["billing", "technical", "sales"])}
LOGITS = {"Classify the intent.": [5.0, 1.0, 0.0]}


def null_calls(backend):
    return [c for c in backend.calls if "N/A" in c[0]]


def test_the_null_prior_is_computed_once_and_reused_across_calls():
    # It depends on the question, not the state, so recomputing it per call doubles
    # the cost of contextual calibration in a loop for no benefit.
    backend = RecordingBackend(LOGITS)
    client = Decider(backend, calibration=Calibration(contextual=True))
    client.ask({"ticket": "one"}, Q)
    client.ask({"ticket": "two"}, Q)
    client.ask({"ticket": "three"}, Q)
    assert len(null_calls(backend)) == 1


def test_the_real_state_is_still_scored_on_every_call():
    backend = RecordingBackend(LOGITS)
    client = Decider(backend, calibration=Calibration(contextual=True))
    client.ask({"ticket": "one"}, Q)
    client.ask({"ticket": "two"}, Q)
    real = [c for c in backend.calls if "N/A" not in c[0]]
    assert len(real) == 2


def test_a_different_question_gets_its_own_null_prior():
    backend = RecordingBackend({**LOGITS, "Is it urgent?": [1.0, 0.0, 0.0]})
    client = Decider(backend, calibration=Calibration(contextual=True))
    client.ask({"ticket": "one"}, Q)
    client.ask({"ticket": "one"}, {"urgent": ChoiceQuestion("Is it urgent?", ["y", "n"])})
    assert len(null_calls(backend)) == 2


def test_caching_does_not_change_the_answer():
    backend = RecordingBackend(LOGITS)
    client = Decider(backend, calibration=Calibration(contextual=True))
    first = client.ask({"ticket": "one"}, Q).answers["intent"].probabilities
    second = client.ask({"ticket": "one"}, Q).answers["intent"].probabilities
    assert first == pytest.approx(second)


def test_the_cache_is_per_client_not_global():
    a = RecordingBackend(LOGITS)
    b = RecordingBackend(LOGITS)
    Decider(a, calibration=Calibration(contextual=True)).ask({"t": "x"}, Q)
    Decider(b, calibration=Calibration(contextual=True)).ask({"t": "x"}, Q)
    assert len(null_calls(a)) == 1
    assert len(null_calls(b)) == 1
