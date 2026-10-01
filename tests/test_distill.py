import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from typed_decisions.distill import Head, TrainConfig, state_text, targets, train_head
from typed_decisions.traffic import Example, family_key
from typed_decisions.types import ChoiceQuestion, NoulQuestion, ScoreQuestion

ROUTE = ChoiceQuestion("Which team?", ["billing", "technical"])
WORDS = ["refund", "charge", "invoice", "crash", "error", "bug", "ticket", "the", "my"]


def example(ticket, teacher, outcome=None, question=ROUTE):
    return Example("r", family_key(question), question, {"ticket": ticket}, teacher, outcome)


BILLING = ["refund my charge", "invoice charge", "my refund", "the invoice", "charge the invoice"]
TECH = ["crash error", "the bug", "my crash", "error bug", "bug crash"]


def dataset():
    return ([example(t, [0.9, 0.1]) for t in BILLING] +
            [example(t, [0.1, 0.9]) for t in TECH])


def test_state_text_matches_the_teacher_prompt_body():
    assert state_text({"b": 1, "a": 2}) == '{\n  "a": 2,\n  "b": 1\n}'
    assert state_text("raw") == "raw"


def test_targets_use_the_teacher_when_no_outcome_is_known():
    assert targets(example("x", [0.7, 0.3]), mix=1.0) == [0.7, 0.3]


def test_a_known_outcome_replaces_the_teacher_at_full_mix():
    assert targets(example("x", [0.7, 0.3], outcome=1), mix=1.0) == [0.0, 1.0]


def test_partial_mix_blends_outcome_and_teacher():
    assert targets(example("x", [0.7, 0.3], outcome=1), mix=0.5) == pytest.approx([0.35, 0.65])


def test_refuses_to_train_one_head_on_two_families(tiny_base):
    other = ChoiceQuestion("Which queue?", ["billing", "technical"])
    with pytest.raises(ValueError, match="one family"):
        train_head([example("a", [0.5, 0.5]), example("b", [0.5, 0.5], question=other)],
                   base=tiny_base, device="cpu", log=lambda *_: None)


def test_refuses_an_empty_training_set(tiny_base):
    with pytest.raises(ValueError):
        train_head([], base=tiny_base, device="cpu")


@pytest.fixture(scope="module")
def trained(tiny_base):
    return train_head(dataset(), base=tiny_base, device="cpu", log=lambda *_: None,
                      config=TrainConfig(epochs=30, lr=3e-3, batch_size=4))


def test_student_learns_to_separate_the_teachers_classes(trained):
    billing, tech = trained.predict_proba([{"ticket": "refund charge"}, {"ticket": "crash bug"}])
    assert billing[0] > 0.5 > tech[0]


def test_answer_has_the_same_typed_shape_as_the_teacher(trained):
    answer = trained.answer({"ticket": "invoice refund"})
    assert answer.choice == "billing"
    assert set(answer.probabilities) == {"billing", "technical"}


def test_temperature_is_fitted_on_outcomes_only(trained):
    with pytest.raises(ValueError, match="outcomes"):
        trained.fit_temperature(dataset())
    labelled = [example("refund", [0.5, 0.5], 0), example("crash", [0.5, 0.5], 1)]
    fit = trained.fit_temperature(labelled)
    assert fit.temperature > 0
    assert trained.temperature == (fit.temperature if fit.trustworthy else 1.0)


def test_a_separable_calibration_set_does_not_sharpen_the_head(trained):
    # the trained student gets all of these right: no information about T
    rows = [example(t, [0.5, 0.5], 0) for t in BILLING] + [example(t, [0.5, 0.5], 1) for t in TECH]
    fit = trained.fit_temperature(rows)
    assert fit.separable and not fit.trustworthy
    assert trained.temperature == 1.0


def test_save_and_load_round_trip_the_predictions(trained, tmp_path):
    trained.save(tmp_path / "head")
    loaded = Head.load(tmp_path / "head", device="cpu")
    assert loaded.question == ROUTE and loaded.temperature == trained.temperature
    states = [{"ticket": "refund"}, {"ticket": "bug"}]
    for a, b in zip(trained.predict_proba(states), loaded.predict_proba(states)):
        assert a == pytest.approx(b, abs=1e-5)


@pytest.mark.parametrize("question, teacher", [
    (NoulQuestion("Urgent?"), [0.8, 0.2]),
    (ScoreQuestion("Severity?", {"low": 1.0, "mid": 3.0, "high": 5.0}), [0.2, 0.3, 0.5]),
])
def test_noul_and_score_families_train_and_answer(tiny_base, question, teacher):
    head = train_head([example("refund", teacher, question=question)], base=tiny_base,
                      device="cpu", log=lambda *_: None, config=TrainConfig(epochs=1))
    answer = head.answer({"ticket": "refund"})
    if isinstance(question, NoulQuestion):
        assert 0.0 < answer.noul < 1.0
    else:
        assert 1.0 <= answer.score <= 5.0


# --- never train a head on its own answers ------------------------------------------

def head_served(outcome=None):
    return Example("h", family_key(ROUTE), ROUTE, {"ticket": "refund"}, [0.99, 0.01], outcome,
                   source="head:x")


def test_a_head_served_row_without_an_outcome_is_not_a_target():
    with pytest.raises(ValueError, match="outcome"):
        targets(head_served(), mix=0.0)


def test_a_head_served_row_with_an_outcome_trains_on_the_outcome_only():
    # even at mix=0 (teacher-only), the head's own distribution must not leak in
    assert targets(head_served(outcome=1), mix=0.0) == [0.0, 1.0]


def test_trainable_drops_unlabelled_head_rows_and_keeps_everything_else():
    from typed_decisions.distill import trainable
    rows = [head_served(), head_served(outcome=0), example("refund", [0.9, 0.1])]
    assert [r.source for r in trainable(rows)] == ["head:x", "teacher"]


# --- promotion gate -----------------------------------------------------------------

def labelled(n):
    rows = []
    for i in range(n):
        text, outcome = (BILLING[i % 5], 0) if i % 2 == 0 else (TECH[i % 5], 1)
        rows.append(Example(f"req-{i}", family_key(ROUTE), ROUTE, {"ticket": text},
                            [0.5, 0.5], outcome))
    return rows


def test_split_is_deterministic_disjoint_and_holds_out_only_labelled_teacher_rows():
    from typed_decisions.distill import split
    rows = labelled(200) + [example("refund", [0.9, 0.1])]
    train, temp, ev = split(rows, 0.3)
    assert split(rows, 0.3) == (train, temp, ev)
    ids = lambda rs: {r.request_id for r in rs}
    assert not (ids(train) & ids(temp)) and not (ids(train) & ids(ev)) and not (ids(temp) & ids(ev))
    assert 30 < len(temp) + len(ev) < 90
    assert all(r.outcome is not None for r in temp + ev)


def test_promote_gates_the_heads_own_predictions_against_the_logged_teacher(trained):
    from typed_decisions.distill import promote
    from typed_decisions.gate import GatePolicy
    rows = labelled(60)    # teacher is a coin flip here; the trained student separates
    verdict = promote(trained, rows, GatePolicy(min_eval=30))
    assert verdict["promoted"], verdict["reasons"]
    assert verdict["source"] == "holdout" and verdict["version"] == trained.version
    assert not promote(trained, rows, GatePolicy(min_eval=100))["promoted"]


def test_promote_can_judge_logged_shadow_answers_instead_of_recomputing(trained):
    from typed_decisions.distill import promote
    from typed_decisions.gate import GatePolicy
    rows = labelled(60)
    wrong = [[0.0, 1.0] if r.outcome == 0 else [1.0, 0.0] for r in rows]
    verdict = promote(trained, rows, GatePolicy(), student=wrong, source="shadow")
    assert not verdict["promoted"] and verdict["source"] == "shadow"


def test_cli_trains_gates_and_writes_a_loadable_head(tiny_base, tmp_path):
    from typed_decisions.distill import main
    from typed_decisions.routing import load_heads
    from typed_decisions.traffic import TrafficLog
    log = TrafficLog(tmp_path / "t.jsonl")
    for i in range(60):
        billing = i % 2 == 0
        text = (BILLING if billing else TECH)[i % 5]
        answer = {"route": {"choice": None, "confidence": 0, "probabilities":
                            {"billing": 0.5, "technical": 0.5}}}
        rid = log.record_ask({"ticket": text}, {"route": ROUTE}, {"answers": answer}, "t")
        log.record_outcome(rid, {"route": "billing" if billing else "technical"})
    main(["train", "--log", str(log.path), "--family", family_key(ROUTE), "--out",
          str(tmp_path / "heads"), "--base", tiny_base, "--device", "cpu", "--epochs", "30",
          "--holdout", "0.5", "--min-eval", "5"])
    assert served_version_is_stable(tmp_path / "heads")
    served, skipped = load_heads(tmp_path / "heads", device="cpu", include_unpromoted=True)
    assert len(served) == 1 and served[0].question == ROUTE
    assert "student_brier" in served[0].evaluation


def served_version_is_stable(heads_dir):
    from typed_decisions.routing import load_heads
    first, _ = load_heads(heads_dir, device="cpu", include_unpromoted=True)
    again, _ = load_heads(heads_dir, device="cpu", include_unpromoted=True)
    return first[0].version == again[0].version


def test_shadow_cli_judges_only_this_head_versions_logged_answers(trained, tmp_path):
    import json
    from typed_decisions.distill import main
    from typed_decisions.traffic import TrafficLog
    trained.save(tmp_path / "head")
    from typed_decisions.distill import Head as H
    head = H.load(tmp_path / "head", device="cpu")
    log = TrafficLog(tmp_path / "t.jsonl")
    teacher = {"billing": 0.5, "technical": 0.5}
    for i in range(60):
        billing = i % 2 == 0
        right = [0.95, 0.05] if billing else [0.05, 0.95]
        version = head.version if i < 50 else "someone-else"
        rid = log.record_ask({"ticket": "x"}, {"route": ROUTE}, {
            "answers": {"route": {"choice": None, "confidence": 0, "probabilities": teacher}},
            "shadow": {"route": {"head": head.label, "version": version, "probabilities": right}},
        }, "t")
        log.record_outcome(rid, {"route": "billing" if billing else "technical"})
    main(["shadow", "--log", str(log.path), "--head", str(tmp_path / "head"), "--write"])
    verdict = json.loads((tmp_path / "head" / "head.json").read_text())["evaluation"]
    assert verdict["source"] == "shadow" and verdict["n_eval"] == 50
    assert verdict["promoted"], verdict["reasons"]
