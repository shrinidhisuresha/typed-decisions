import json
from typed_decisions.client import Decider
from typed_decisions.types import ChoiceQuestion, NoulQuestion, ScoreQuestion
from tests.doubles import RecordingBackend

STATE = {"ticket": "charged twice"}
QUESTIONS = {
    "intent": ChoiceQuestion("Classify the intent.", ["billing", "sales"]),
    "urgent": NoulQuestion("Is this urgent?"),
    "severity": ScoreQuestion("Rate severity.", {"low": 1.0, "high": 5.0}),
}
LOGITS = {
    "Classify the intent.": [5.0, 0.0],
    "Is this urgent?": [2.0, 1.0],
    "Rate severity.": [0.0, 10.0],
}


def result():
    return Decider(RecordingBackend(LOGITS)).ask(STATE, QUESTIONS)


def test_top_level_shape_is_answers_and_usage():
    body = result().to_dict()
    assert set(body) == {"answers", "usage"}
    assert set(body["usage"]) == {"input_tokens", "output_tokens"}


def test_choice_answer_carries_choice_confidence_and_probabilities():
    answer = result().to_dict()["answers"]["intent"]
    assert set(answer) == {"choice", "confidence", "probabilities"}
    assert answer["choice"] == "billing"


def test_noul_answer_carries_only_the_noul_value():
    answer = result().to_dict()["answers"]["urgent"]
    assert set(answer) == {"noul"}


def test_score_answer_carries_score_legend_probabilities_and_confidence():
    answer = result().to_dict()["answers"]["severity"]
    assert set(answer) == {"score", "legend", "probabilities", "confidence"}


def test_whole_response_is_json_serialisable():
    json.dumps(result().to_dict())
