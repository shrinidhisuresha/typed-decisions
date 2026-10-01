import pytest

from typed_decisions.traffic import TrafficLog, examples, family_key, teacher_distribution
from typed_decisions.types import ChoiceQuestion, NoulQuestion, ScoreQuestion

ROUTE = ChoiceQuestion("Which team?", ["billing", "technical", "sales"])
URGENT = NoulQuestion("Urgent?")
SEV = ScoreQuestion("Severity?", {"low": 1.0, "high": 5.0})

RESPONSE = {"answers": {
    "route": {"choice": "billing", "probabilities": {"billing": 0.7, "technical": 0.2, "sales": 0.1},
              "confidence": 0.5},
    "urgent": {"noul": 0.8},
    "sev": {"score": 4.2, "legend": {"low": 1.0, "high": 5.0},
            "probabilities": {"low": 0.2, "high": 0.8}, "confidence": 0.6},
}, "usage": {"input_tokens": 10, "output_tokens": 0}}


def logged(tmp_path, outcomes=None):
    log = TrafficLog(tmp_path / "traffic.jsonl")
    rid = log.record_ask({"ticket": "charged twice"},
                         {"route": ROUTE, "urgent": URGENT, "sev": SEV}, RESPONSE, "m")
    if outcomes:
        log.record_outcome(rid, outcomes)
    return log, rid


def test_family_key_is_stable_and_changes_when_the_question_is_reworded():
    assert family_key(ROUTE) == family_key(ChoiceQuestion("Which team?", ["billing", "technical", "sales"]))
    assert family_key(ROUTE) != family_key(ChoiceQuestion("Which queue?", ["billing", "technical", "sales"]))
    assert family_key(ROUTE) != family_key(ChoiceQuestion("Which team?", ["technical", "billing", "sales"]))


def test_a_score_legend_keeps_its_order_through_the_log(tmp_path):
    log, _ = logged(tmp_path)
    sev = [r for r in examples(log) if r.family == family_key(SEV)][0]
    assert sev.options == ["low", "high"]
    assert sev.teacher == [0.2, 0.8]


def test_teacher_distribution_is_aligned_to_declared_options():
    assert teacher_distribution(ROUTE, RESPONSE["answers"]["route"]) == [0.7, 0.2, 0.1]
    assert teacher_distribution(URGENT, RESPONSE["answers"]["urgent"]) == pytest.approx([0.8, 0.2])
    assert teacher_distribution(SEV, RESPONSE["answers"]["sev"]) == [0.2, 0.8]


def test_every_question_in_an_ask_becomes_one_example(tmp_path):
    log, rid = logged(tmp_path)
    rows = examples(log)
    assert len(rows) == 3
    assert {r.request_id for r in rows} == {rid}
    assert all(r.outcome is None for r in rows)
    assert rows[0].state == {"ticket": "charged twice"}


def test_outcomes_join_to_their_ask_by_request_id(tmp_path):
    log, _ = logged(tmp_path, {"route": "technical", "urgent": False, "sev": "high"})
    by_family = {r.family: r for r in examples(log)}
    assert by_family[family_key(ROUTE)].outcome == 1
    assert by_family[family_key(URGENT)].outcome == 1   # False -> "no"
    assert by_family[family_key(SEV)].outcome == 1


def test_the_latest_outcome_wins(tmp_path):
    log, rid = logged(tmp_path, {"route": "technical"})
    log.record_outcome(rid, {"route": "sales"})
    route = [r for r in examples(log) if r.family == family_key(ROUTE)][0]
    assert route.outcome == 2


def test_filters_to_one_family(tmp_path):
    log, _ = logged(tmp_path)
    rows = examples(log, family=family_key(URGENT))
    assert len(rows) == 1 and rows[0].question == URGENT


def test_outcomes_for_unknown_requests_are_skipped(tmp_path):
    log, _ = logged(tmp_path)
    log.record_outcome("rotated-away", {"route": "billing"})
    assert len(examples(log)) == 3


def test_an_invalid_label_raises_rather_than_training_on_garbage(tmp_path):
    log, _ = logged(tmp_path, {"route": "marketing"})
    with pytest.raises(ValueError, match="marketing"):
        examples(log)


def test_an_empty_outcome_is_refused_at_write_time(tmp_path):
    log, rid = logged(tmp_path)
    with pytest.raises(ValueError):
        log.record_outcome(rid, {})


def test_a_missing_log_file_has_no_examples(tmp_path):
    assert examples(TrafficLog(tmp_path / "nope.jsonl")) == []


def test_served_by_is_recorded_and_becomes_the_example_source(tmp_path):
    log = TrafficLog(tmp_path / "t.jsonl")
    response = {**RESPONSE, "served_by": {"route": "head:abc", "urgent": "teacher", "sev": "teacher"}}
    log.record_ask({"ticket": "x"}, {"route": ROUTE, "urgent": URGENT, "sev": SEV}, response, "m")
    sources = {e.question.instructions: e.source for e in examples(log)}
    assert sources == {"Which team?": "head:abc", "Urgent?": "teacher", "Severity?": "teacher"}


def test_asks_logged_before_routing_existed_default_to_the_teacher(tmp_path):
    log, _ = logged(tmp_path)
    assert {e.source for e in examples(log)} == {"teacher"}
