import math
import pytest

from typed_decisions.calibration.basecal import pool
from typed_decisions.client import Calibration, Decider
from typed_decisions.types import ChoiceQuestion
from tests.doubles import FakeTokenizer, PriorOnlyBackend, RecordingBackend

QUESTION = {"route": ChoiceQuestion("Classify the intent.", ["billing", "technical", "sales"])}
STATE = {"ticket": "charged twice"}


def test_weight_zero_is_instruct_alone_and_one_is_base_alone():
    assert pool([0.9, 0.1], [0.4, 0.6], 0.0) == [0.9, 0.1]
    assert pool([0.9, 0.1], [0.4, 0.6], 1.0) == [0.4, 0.6]


def test_pool_is_a_normalised_geometric_mix():
    got = pool([0.9, 0.1], [0.4, 0.6], 0.5)
    a, b = math.sqrt(0.9 * 0.4), math.sqrt(0.1 * 0.6)
    assert got == pytest.approx([a / (a + b), b / (a + b)])
    assert sum(got) == pytest.approx(1.0)


def test_pooling_with_a_less_confident_base_softens_an_overconfident_instruct():
    instruct, base = [0.98, 0.01, 0.01], [0.5, 0.3, 0.2]
    pooled = pool(instruct, base, 0.5)
    assert pooled.index(max(pooled)) == 0
    assert pooled[0] < instruct[0]


def test_a_zero_on_either_side_stays_finite():
    got = pool([1.0, 0.0], [0.5, 0.5], 0.5)
    assert all(math.isfinite(p) for p in got) and got[0] > 0.99


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_rejects_weights_outside_the_unit_interval(weight):
    with pytest.raises(ValueError):
        pool([0.5, 0.5], [0.5, 0.5], weight)


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        pool([0.5, 0.5], [0.2, 0.3, 0.5], 0.5)


def test_basecal_without_a_reference_is_refused():
    with pytest.raises(ValueError, match="reference"):
        Decider(RecordingBackend({}), calibration=Calibration(basecal=0.5))


def test_reference_with_a_different_tokenizer_is_refused():
    class Other(FakeTokenizer):
        def encode(self, text):
            return [9] if len(text) == 2 else super().encode(text)

    reference = PriorOnlyBackend([0.0, 0.0, 0.0])
    reference.tokenizer = Other()
    with pytest.raises(ValueError, match="Base twin"):
        Decider(RecordingBackend({}), calibration=Calibration(basecal=0.5), reference=reference)


def test_client_scores_both_models_on_identical_branches_and_pools():
    instruct = RecordingBackend({"Classify the intent.": [6.0, 0.0, 0.0]})
    base = PriorOnlyBackend([0.0, 0.0, 0.0])  # uniform: carries no information
    client = Decider(instruct, calibration=Calibration(basecal=0.5), reference=base)
    answer = client.ask(STATE, QUESTION).answers["route"]

    assert [b.text for b in instruct.calls[0][1]] == [b.text for b in base.calls[0][1]]
    assert instruct.calls[0][0] == base.calls[0][0]
    alone = Decider(instruct).ask(STATE, QUESTION).answers["route"]
    assert answer.choice == "billing"
    assert answer.probabilities["billing"] < alone.probabilities["billing"]


def test_reference_is_ignored_when_basecal_is_zero():
    base = PriorOnlyBackend([0.0, 0.0, 0.0])
    Decider(RecordingBackend({"Classify the intent.": [1.0, 0.0, 0.0]}), reference=base).ask(
        STATE, QUESTION)
    assert base.calls == []


def test_contextual_null_prior_is_cached_separately_per_model():
    instruct = RecordingBackend({"Classify the intent.": [2.0, 0.0, 0.0]})
    base = PriorOnlyBackend([1.0, 0.0, 0.0])
    client = Decider(instruct, calibration=Calibration(basecal=0.5, contextual=True), reference=base)
    client.ask(STATE, QUESTION)
    client.ask({"ticket": "other"}, QUESTION)
    # each model: 2 real calls + 1 null-prior call, not 2
    assert len(instruct.calls) == 3 and len(base.calls) == 3
