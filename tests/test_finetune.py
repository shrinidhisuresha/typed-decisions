import math
from collections import Counter

import pytest

torch = pytest.importorskip("torch")

from tests.doubles import FakeTokenizer
from typed_decisions.benchmarks import Sample
from typed_decisions.calibration.contextual import NULL_STATE
from typed_decisions.finetune.data import build, encode
from typed_decisions.finetune.loss import (
    log_score,
    ranked_probability_score,
    restricted_log_probs,
)
from typed_decisions.prompt import declared_options, render_branch, render_prefix
from typed_decisions.types import ChoiceQuestion, NoulQuestion

LABELS = [f"intent {i}" for i in range(20)]
SAMPLES = [Sample(f"message {i}", LABELS[i % 20]) for i in range(400)]


def test_restricted_softmax_ignores_padding_and_other_tokens():
    logits = torch.tensor([[0.0, 5.0, 1.0, 1.0, 9.0]])
    logp = restricted_log_probs(logits, torch.tensor([[2, 3, 0]]), torch.tensor([[True, True, False]]))
    assert torch.allclose(logp.exp()[0, :2], torch.tensor([0.5, 0.5]))
    assert logp.exp()[0, 2] == 0


def test_log_score_is_minimised_by_the_target_distribution():
    target = torch.tensor([[0.7, 0.3]])
    mask = torch.ones(1, 2, dtype=torch.bool)

    def score(p):
        return float(log_score(torch.log(torch.tensor([[p, 1 - p]])), target, mask))

    assert score(0.7) < score(0.6) and score(0.7) < score(0.8)


def test_rps_charges_distance_on_ordinal_answers():
    truth = torch.tensor([[0.0, 0.0, 1.0]])
    near = ranked_probability_score(torch.tensor([[0.0, 1.0, 0.0]]), truth)
    far = ranked_probability_score(torch.tensor([[1.0, 0.0, 0.0]]), truth)
    assert near < far


def test_build_puts_the_label_in_every_position():
    rows = [r for r in build(SAMPLES, LABELS, null_share=0.0, noul_share=0.0)]
    positions = Counter(r.target.index(1.0) for r in rows if len(r.target) == 3)
    assert set(positions) == {0, 1, 2}


def test_targets_line_up_with_declared_options():
    for row in build(SAMPLES, LABELS):
        options = declared_options(row.question)
        assert len(options) == len(row.target)
        assert math.isclose(sum(row.target), 1.0)
        if isinstance(row.question, ChoiceQuestion) and row.state != NULL_STATE:
            assert options[row.target.index(1.0)] == next(
                s.label for s in SAMPLES if s.state == row.state)


def test_null_rows_target_the_uniform_prior():
    nulls = [r for r in build(SAMPLES, LABELS, null_share=1.0) if r.state == NULL_STATE]
    assert nulls
    for r in nulls:
        assert all(math.isclose(t, 1 / len(r.target)) for t in r.target)


def test_noul_rows_are_balanced():
    rows = [r for r in build(SAMPLES, LABELS, noul_share=1.0, null_share=0.0)]
    assert all(isinstance(r.question, NoulQuestion) for r in rows)
    yes = sum(r.target == (1.0, 0.0) for r in rows) / len(rows)
    assert 0.4 < yes < 0.6


def test_encode_renders_the_served_prompt():
    row = build(SAMPLES[:1], LABELS, null_share=0.0, noul_share=0.0)[0]
    tok = FakeTokenizer()
    enc = encode(row, tok)
    assert enc.input_ids == tok.encode(render_prefix(row.state) + render_branch(row.question))
    assert len(enc.label_ids) == len(row.target) == len(set(enc.label_ids))


def test_negated_noul_flips_which_label_is_yes():
    import random
    from typed_decisions.finetune.data import _native_noul
    rng = random.Random(0)
    rows = [_native_noul("x", "spam", ["spam", "ham"], rng, ("Is this spam?",), ("Is this ham?",))
            for _ in range(200)]
    for r in rows:
        expected = (1.0, 0.0) if r.question.instructions == "Is this spam?" else (0.0, 1.0)
        assert r.target == expected
    assert {r.question.instructions for r in rows} == {"Is this spam?", "Is this ham?"}


def test_score_rows_keep_ordinal_values_when_shuffled():
    import random
    from typed_decisions.finetune.data import _score
    levels = ["low", "mid", "high"]
    rng = random.Random(1)
    for _ in range(50):
        row = _score("x", "mid", levels, rng, ("How much?",))
        assert row.question.criteria == {"low": 1.0, "mid": 2.0, "high": 3.0} | row.question.criteria
        assert declared_options(row.question)[row.target.index(1.0)] == "mid"


def test_ordinal_rps_reads_levels_in_natural_order_not_display_order():
    from typed_decisions.finetune.loss import ordinal_rps
    # displayed as [high, low, mid]; natural order is low(1), mid(2), high(0)
    probs = torch.tensor([[0.0, 0.0, 1.0]])            # all mass on "mid"
    target = torch.tensor([[1.0, 0.0, 0.0]])           # truth "high"
    near = ordinal_rps(probs.clamp_min(1e-9).log(), target, [(1, 2, 0)])
    far = ordinal_rps(torch.tensor([[0.0, 1.0, 0.0]]).clamp_min(1e-9).log(), target, [(1, 2, 0)])
    assert near < far
    assert ordinal_rps(probs.log(), target, [()]) == 0


def test_encode_marks_score_rows_ordinal():
    import random
    from typed_decisions.finetune.data import _score
    row = _score("x", "mid", ["low", "mid", "high"], random.Random(3), ("How much?",))
    enc = encode(row, FakeTokenizer())
    options = declared_options(row.question)
    assert [options[i] for i in enc.ordinal] == ["low", "mid", "high"]


def test_confidence_penalty_marks_only_noul_and_score_rows():
    import random
    from typed_decisions.finetune.data import _choice, _noul, _score
    rng = random.Random(0)
    tok = FakeTokenizer()
    assert not encode(_choice("x", LABELS[0], LABELS, rng, (3, 5)), tok).penalised
    assert encode(_noul("x", LABELS[0], LABELS, rng), tok).penalised
    assert encode(_score("x", "mid", ["low", "mid", "high"], rng, ("How?",)), tok).penalised


def test_entropy_is_highest_for_the_uniform_distribution_and_ignores_padding():
    from typed_decisions.finetune.loss import entropy
    mask = torch.tensor([[True, True, False]])
    flat = entropy(torch.log(torch.tensor([[0.5, 0.5, 1e-9]])), mask)
    sharp = entropy(torch.log(torch.tensor([[0.9, 0.1, 1e-9]])), mask)
    assert math.isclose(float(flat), math.log(2), rel_tol=1e-5)
    assert sharp < flat
