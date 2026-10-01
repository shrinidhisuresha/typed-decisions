import pytest

from typed_decisions.client import Calibration, Decider
from typed_decisions.retrieval import shortlist
from typed_decisions.types import ChoiceQuestion, NoulQuestion
from tests.doubles import RecordingBackend

OPTIONS = [f"team {i}" for i in range(40)]
STATE = {"ticket": "team 7 and team 31 please"}


class KeywordRetriever:
    """Scores an option 1 if its text appears in the query, else by closeness to 20."""

    def __init__(self):
        self.calls = []

    def rank(self, query, options, task):
        self.calls.append((query, list(options), task))
        import re
        hit = lambda o: re.search(rf"\b{re.escape(o)}\b", query) is not None
        return [1.0 if hit(o) else -abs(int(o.split()[1]) - 20) / 100 for o in options]


class LastBranch(RecordingBackend):
    def __init__(self):
        super().__init__({})

    def score(self, prefix, branches):
        self.calls.append((prefix, list(branches)))
        # favour the option shown second, so the answer depends on the listing
        return [[0.0, 3.0] + [0.0] * (b.n_options - 2) if b.n_options > 1 else [0.0]
                for b in branches]


def test_shortlist_keeps_the_top_k_in_original_order():
    assert shortlist([0.1, 0.9, 0.5, 0.7], 2) == [1, 3]
    assert shortlist([0.5, 0.5, 0.5], 2) == [0, 1]           # ties broken by position
    with pytest.raises(ValueError):
        shortlist([1.0], 0)


def test_a_long_choice_is_narrowed_before_scoring_and_expanded_after():
    retriever, backend = KeywordRetriever(), LastBranch()
    answer = Decider(backend, retriever=retriever, shortlist_k=4).ask(
        STATE, {"q": ChoiceQuestion("Which team?", OPTIONS)}).answers["q"]
    listed = backend.calls[0][1][0].text
    assert "A) team 7" in listed and "team 31" in listed and "team 0" not in listed
    assert len(backend.calls[0][1][0].token_ids) == 4          # letters, not option text
    assert set(answer.probabilities) == set(OPTIONS)            # full list back
    assert sum(1 for p in answer.probabilities.values() if p > 0) == 4
    assert answer.probabilities["team 0"] == 0.0
    assert answer.choice in {"team 7", "team 19", "team 20", "team 31"}


def test_the_retriever_sees_the_state_body_and_the_question():
    retriever = KeywordRetriever()
    Decider(LastBranch(), retriever=retriever, shortlist_k=4).ask(
        STATE, {"q": ChoiceQuestion("Which team?", OPTIONS)})
    query, options, task = retriever.calls[0]
    assert '"ticket": "team 7 and team 31 please"' in query
    assert options == OPTIONS and "Which team?" in task


def test_short_choices_and_other_types_are_not_retrieved():
    retriever = KeywordRetriever()
    Decider(LastBranch(), retriever=retriever, shortlist_k=4).ask(
        STATE, {"small": ChoiceQuestion("Which?", OPTIONS[:26]), "u": NoulQuestion("Urgent?")})
    assert retriever.calls == []


def test_retrieve_above_lowers_the_trigger():
    retriever = KeywordRetriever()
    Decider(LastBranch(), retriever=retriever, shortlist_k=4, retrieve_above=5).ask(
        STATE, {"q": ChoiceQuestion("Which?", OPTIONS[:10])})
    assert len(retriever.calls) == 1


def test_without_a_retriever_long_choices_fall_back_to_option_text():
    backend = LastBranch()
    Decider(backend).ask(STATE, {"q": ChoiceQuestion("Which?", OPTIONS)})
    assert backend.calls[0][1][0].continuations


def test_permutations_and_contextual_run_on_the_shortlist():
    backend = LastBranch()
    Decider(backend, retriever=KeywordRetriever(), shortlist_k=4,
        calibration=Calibration(permutations=4, contextual=True)).ask(
        STATE, {"q": ChoiceQuestion("Which team?", OPTIONS)})
    real, null = backend.calls[0][1], backend.calls[1][1]
    assert len(real) == 4 and len(null) == 4                     # 4 rotations, then the prior
    assert all(len(b.token_ids) == 4 for b in real + null)


def test_shortlist_k_is_validated():
    with pytest.raises(ValueError):
        Decider(LastBranch(), shortlist_k=1)


def test_the_embedding_model_is_not_loaded_until_first_used():
    from typed_decisions.retrieval import EmbeddingRetriever
    r = EmbeddingRetriever("some/model-that-is-never-downloaded")
    assert r._loaded is False and not hasattr(r, "model")
