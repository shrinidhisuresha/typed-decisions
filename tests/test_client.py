import pytest
from typed_decisions.client import Decider
from typed_decisions.types import ChoiceQuestion, NoulQuestion, ScoreQuestion
from tests.doubles import RecordingBackend

STATE = {"ticket": "charged twice", "tier": "enterprise"}

QUESTIONS = {
    "intent": ChoiceQuestion("Classify the intent.", ["billing", "technical", "sales"]),
    "urgent": NoulQuestion("Is this urgent?"),
    "severity": ScoreQuestion("Rate severity.", {"low": 1.0, "medium": 3.0, "high": 5.0}),
}

LOGITS = {
    "Classify the intent.": [5.0, 0.0, 0.0],
    "Is this urgent?": [2.0, 1.0],
    "Rate severity.": [0.0, 0.0, 10.0],
}


def make_client():
    return Decider(RecordingBackend(LOGITS))


def test_returns_one_answer_per_question_keyed_by_name():
    result = make_client().ask(STATE, QUESTIONS)
    assert set(result.answers) == {"intent", "urgent", "severity"}


def test_choice_answer_uses_the_option_text_not_the_letter():
    result = make_client().ask(STATE, QUESTIONS)
    assert result.answers["intent"].choice == "billing"


def test_noul_answer_is_a_probability():
    result = make_client().ask(STATE, QUESTIONS)
    assert 0.0 < result.answers["urgent"].noul < 1.0


def test_score_answer_is_an_expectation_over_the_legend():
    result = make_client().ask(STATE, QUESTIONS)
    assert result.answers["severity"].score == pytest.approx(5.0, abs=0.01)


def test_no_output_tokens_are_ever_generated():
    # The core economic claim: we read logits, we do not decode.
    result = make_client().ask(STATE, QUESTIONS)
    assert result.usage.output_tokens == 0


def test_input_tokens_count_the_prefix_once_not_once_per_question():
    backend = RecordingBackend(LOGITS)
    one = Decider(backend).ask(STATE, {"intent": QUESTIONS["intent"]}).usage.input_tokens
    three = Decider(RecordingBackend(LOGITS)).ask(STATE, QUESTIONS).usage.input_tokens
    prefix_tokens = len(backend.calls[0][0]) * 1  # FakeTokenizer: 1 token per char
    # Adding two more questions must cost far less than re-sending the whole state.
    assert three - one < prefix_tokens


def test_all_questions_are_scored_against_a_single_shared_prefix():
    backend = RecordingBackend(LOGITS)
    Decider(backend).ask(STATE, QUESTIONS)
    assert len(backend.calls) == 1, "state must be submitted once, not once per question"
    prefix, branches = backend.calls[0]
    assert len(branches) == 3
    assert "charged twice" in prefix


def test_empty_question_set_is_rejected():
    with pytest.raises(ValueError, match="at least one question"):
        make_client().ask(STATE, {})
