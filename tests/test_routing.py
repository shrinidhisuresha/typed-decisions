import pytest

from typed_decisions.client import Decider
from typed_decisions.routing import Router
from typed_decisions.types import ChoiceQuestion, NoulQuestion
from tests.doubles import FakeHead, RecordingBackend

ROUTE = ChoiceQuestion("Classify the intent.", ["billing", "technical", "sales"])
URGENT = NoulQuestion("Is this urgent?")
LOGITS = {"Classify the intent.": [5.0, 0.0, 0.0], "Is this urgent?": [2.0, 1.0]}
STATE = {"ticket": "charged twice"}


def make(heads=(), min_confidence=0.0):
    backend = RecordingBackend(LOGITS)
    return Router(Decider(backend), heads, min_confidence), backend


def test_without_heads_everything_goes_to_the_teacher():
    router, backend = make()
    out = router.ask(STATE, {"route": ROUTE, "urgent": URGENT})
    assert out["served_by"] == {"route": "teacher", "urgent": "teacher"}
    assert out["answers"]["route"]["choice"] == "billing"


def test_a_head_answers_its_family_and_the_teacher_only_sees_the_rest():
    head = FakeHead(ROUTE, [0.1, 0.8, 0.1])
    router, backend = make([head])
    out = router.ask(STATE, {"route": ROUTE, "urgent": URGENT})
    assert out["answers"]["route"]["choice"] == "technical"
    assert out["served_by"]["route"] == f"head:{head.family}"
    assert out["served_by"]["urgent"] == "teacher"
    branch_texts = [b.text for b in backend.calls[0][1]]
    assert all("Classify the intent." not in t for t in branch_texts)


def test_when_heads_answer_everything_the_teacher_is_never_called():
    router, backend = make([FakeHead(ROUTE, [0.8, 0.1, 0.1]), FakeHead(URGENT, [0.9, 0.1])])
    out = router.ask(STATE, {"route": ROUTE, "urgent": URGENT})
    assert backend.calls == []
    assert out["usage"] == {"input_tokens": 0, "output_tokens": 0}
    assert out["answers"]["urgent"]["noul"] == pytest.approx(0.9)


def test_an_unsure_head_defers_to_the_teacher():
    unsure = FakeHead(ROUTE, [0.4, 0.35, 0.25])     # margin 0.05
    router, backend = make([unsure], min_confidence=0.2)
    out = router.ask(STATE, {"route": ROUTE})
    assert out["served_by"]["route"] == "teacher"
    assert unsure.calls == 1 and len(backend.calls) == 1


def test_a_reworded_question_is_a_different_family_and_is_not_served_by_the_head():
    router, backend = make([FakeHead(ROUTE, [0.1, 0.8, 0.1])])
    reworded = ChoiceQuestion("Which team?", ["billing", "technical", "sales"])
    backend.logits_by_marker["Which team?"] = [5.0, 0.0, 0.0]
    assert router.ask(STATE, {"route": reworded})["served_by"]["route"] == "teacher"


def test_answers_keep_the_callers_question_order():
    router, _ = make([FakeHead(URGENT, [0.9, 0.1])])
    out = router.ask(STATE, {"route": ROUTE, "urgent": URGENT})
    assert list(out["answers"]) == ["route", "urgent"]


def test_two_heads_for_one_family_are_refused():
    with pytest.raises(ValueError, match="two heads"):
        make([FakeHead(ROUTE, [1, 0, 0]), FakeHead(ROUTE, [0, 1, 0])])


def test_min_confidence_is_bounded():
    with pytest.raises(ValueError):
        make(min_confidence=1.5)


def test_a_shadow_head_runs_but_the_teacher_answers():
    shadow = FakeHead(ROUTE, [0.1, 0.8, 0.1], promoted=False)
    backend = RecordingBackend(LOGITS)
    router = Router(Decider(backend), shadow_heads=[shadow])
    out = router.ask(STATE, {"route": ROUTE})
    assert out["served_by"]["route"] == "teacher"
    assert out["answers"]["route"]["choice"] == "billing"
    assert out["shadow"]["route"] == {"head": shadow.label, "version": "v1",
                                      "probabilities": [0.1, 0.8, 0.1]}


def test_a_served_head_takes_precedence_over_a_shadow_head_for_its_family():
    served, shadow = FakeHead(ROUTE, [0.8, 0.1, 0.1]), FakeHead(ROUTE, [0.1, 0.8, 0.1])
    router, _ = make([served])
    router = Router(router.client, [served], shadow_heads=[shadow])
    out = router.ask(STATE, {"route": ROUTE})
    assert "shadow" not in out and shadow.calls == 0


def test_no_shadow_key_when_nothing_ran_in_shadow():
    router, _ = make()
    assert "shadow" not in router.ask(STATE, {"route": ROUTE})
