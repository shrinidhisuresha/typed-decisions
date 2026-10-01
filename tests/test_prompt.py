import pytest
from typed_decisions.types import ChoiceQuestion, ScoreQuestion, NoulQuestion
from typed_decisions.prompt import render_prefix, render_branch


STATE = {"ticket": "my card was charged twice", "tier": "enterprise"}

QUESTIONS = {
    "intent": ChoiceQuestion(
        instructions="Classify the support intent.",
        criteria=["billing", "technical", "sales"],
    ),
    "urgent": NoulQuestion(instructions="Is this urgent?"),
    "severity": ScoreQuestion(
        instructions="Rate severity.",
        criteria={"low": 1.0, "medium": 3.0, "high": 5.0},
    ),
}


def test_every_question_shares_a_byte_identical_prefix():
    # This is the invariant the whole cost model rests on: the state is prefilled
    # once and forked, so it must be a strict prefix of every branch.
    prefix = render_prefix(STATE)
    prompts = [prefix + render_branch(q) for q in QUESTIONS.values()]
    assert all(p.startswith(prefix) for p in prompts)


def test_the_state_appears_in_the_shared_prefix_not_the_branch():
    prefix = render_prefix(STATE)
    assert "my card was charged twice" in prefix
    for q in QUESTIONS.values():
        assert "my card was charged twice" not in render_branch(q)


def test_no_question_text_leaks_into_the_shared_prefix():
    prefix = render_prefix(STATE)
    for q in QUESTIONS.values():
        assert q.instructions not in prefix


def test_questions_are_isolated_from_each_other():
    # Each branch must not mention any other question, or answers stop being independent.
    branch = render_branch(QUESTIONS["intent"])
    assert "urgent" not in branch.lower()
    assert "severity" not in branch.lower()


def test_choice_branch_lists_options_against_their_letters():
    branch = render_branch(QUESTIONS["intent"])
    assert "A) billing" in branch
    assert "B) technical" in branch
    assert "C) sales" in branch


def test_score_branch_lists_the_legend_bands():
    branch = render_branch(QUESTIONS["severity"])
    assert "A) low" in branch
    assert "C) high" in branch


def test_noul_branch_offers_yes_and_no():
    branch = render_branch(QUESTIONS["urgent"])
    assert "A) yes" in branch
    assert "B) no" in branch


def test_branch_ends_without_trailing_whitespace_so_the_label_token_is_clean():
    # Labels are scored as " A"; a trailing space in the prompt would corrupt the
    # BPE boundary and score a different token than the one we allocated.
    for q in QUESTIONS.values():
        branch = render_branch(q)
        assert branch == branch.rstrip(), f"branch for {q} ends in whitespace"
