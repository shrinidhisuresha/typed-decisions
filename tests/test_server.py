import json
import threading
from http.server import ThreadingHTTPServer

import httpx
import pytest

from typed_decisions.client import Decider
from typed_decisions.server import make_handler
from typed_decisions.types import ChoiceQuestion, NoulQuestion, ScoreQuestion, question_from_dict
from tests.doubles import RecordingBackend

LOGITS = {
    "Classify the intent.": [5.0, 0.0, 0.0],
    "Is this urgent?": [2.0, 1.0],
    "Rate severity.": [0.0, 0.0, 10.0],
}

QUESTIONS = {
    "intent": {"type": "choice", "instructions": "Classify the intent.",
               "criteria": ["billing", "technical", "sales"]},
    "urgent": {"type": "noul", "instructions": "Is this urgent?"},
    "severity": {"type": "score", "instructions": "Rate severity.",
                 "criteria": {"low": 1, "medium": 3, "high": 5}},
}


@pytest.fixture
def base_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Decider(RecordingBackend(LOGITS)), "fake"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture
def logged_server(tmp_path):
    from typed_decisions.traffic import TrafficLog
    log = TrafficLog(tmp_path / "traffic.jsonl")
    server = ThreadingHTTPServer(("127.0.0.1", 0),
                                 make_handler(Decider(RecordingBackend(LOGITS)), "fake", log))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", log
    server.shutdown()


def ask(url):
    return httpx.post(f"{url}/v1/ask", json={"state": {"ticket": "charged twice"},
                                             "questions": QUESTIONS})


def test_a_logged_ask_returns_a_request_id_and_is_written(logged_server):
    url, log = logged_server
    body = ask(url).json()
    records = list(log.records())
    assert records[0]["kind"] == "ask" and records[0]["id"] == body["request_id"]
    assert records[0]["answers"] == body["answers"]


def test_outcomes_are_recorded_and_join_to_the_ask(logged_server):
    from typed_decisions.traffic import examples
    url, log = logged_server
    rid = ask(url).json()["request_id"]
    r = httpx.post(f"{url}/v1/outcomes", json={"request_id": rid,
                                                "outcomes": {"intent": "sales", "urgent": True}})
    assert r.status_code == 200
    joined = {e.question.instructions: e.outcome for e in examples(log)}
    assert joined == {"Classify the intent.": 2, "Is this urgent?": 0, "Rate severity.": None}


@pytest.mark.parametrize("outcomes", [{"intent": "marketing"}, {"nonexistent": "x"}, {}])
def test_bad_outcomes_for_a_known_request_are_rejected(logged_server, outcomes):
    url, log = logged_server
    rid = ask(url).json()["request_id"]
    r = httpx.post(f"{url}/v1/outcomes", json={"request_id": rid, "outcomes": outcomes})
    assert r.status_code == 400
    assert [rec["kind"] for rec in log.records()] == ["ask"]


def test_outcomes_without_logging_is_a_conflict(base_url):
    r = httpx.post(f"{base_url}/v1/outcomes", json={"request_id": "x", "outcomes": {"a": 1}})
    assert r.status_code == 409


def test_unlogged_ask_carries_no_request_id(base_url):
    assert "request_id" not in ask(base_url).json()


def test_ask_returns_the_answers_and_usage_shape(base_url):
    r = httpx.post(f"{base_url}/v1/ask", json={"state": {"ticket": "charged twice"},
                                                "questions": QUESTIONS})
    assert r.status_code == 200
    body = r.json()
    assert body["answers"]["intent"]["choice"] == "billing"
    assert 0.0 < body["answers"]["urgent"]["noul"] < 1.0
    assert body["answers"]["severity"]["score"] > 4.0
    assert body["usage"]["output_tokens"] == 0


def test_health(base_url):
    assert httpx.get(f"{base_url}/healthz").json() == {"status": "ok", "model": "fake"}


def test_stats_is_null_for_a_backend_that_does_not_count(base_url):
    assert httpx.get(f"{base_url}/v1/stats").json() == {"stats": None}


@pytest.mark.parametrize("payload", [
    {"questions": QUESTIONS},
    {"state": {}, "questions": {}},
    {"state": {}, "questions": {"x": {"type": "choice", "instructions": "?", "criteria": ["one"]}}},
    {"state": {}, "questions": {"x": {"type": "vibes", "instructions": "?"}}},
])
def test_rejects_malformed_requests_with_400(base_url, payload):
    r = httpx.post(f"{base_url}/v1/ask", json=payload)
    assert r.status_code == 400
    assert "error" in r.json()


def test_rejects_non_json_with_400(base_url):
    r = httpx.post(f"{base_url}/v1/ask", content=b"not json",
                   headers={"Content-Type": "application/json"})
    assert r.status_code == 400


def test_unknown_route_is_404(base_url):
    assert httpx.get(f"{base_url}/nope").status_code == 404


def test_question_from_dict_round_trips_every_type():
    assert question_from_dict(QUESTIONS["intent"]) == ChoiceQuestion(
        "Classify the intent.", ["billing", "technical", "sales"])
    assert question_from_dict(QUESTIONS["urgent"]) == NoulQuestion("Is this urgent?")
    assert question_from_dict(QUESTIONS["severity"]) == ScoreQuestion(
        "Rate severity.", {"low": 1.0, "medium": 3.0, "high": 5.0})


def test_basecal_without_a_reference_exits_before_loading_anything(monkeypatch):
    import argparse
    import typed_decisions.server as server
    monkeypatch.setattr(server, "build_backend", lambda *a, **k: object())
    args = argparse.Namespace(backend="sglang", model="m", url=None, device=None, dtype=None,
                              permutations=1, contextual=False, temperature=1.0,
                              basecal=0.5, reference=None, reference_url=None,
                              retriever=None, shortlist_k=10)
    with pytest.raises(SystemExit, match="reference"):
        server.build_client(args)


@pytest.fixture
def routed_server(tmp_path):
    from typed_decisions.traffic import TrafficLog
    from tests.doubles import FakeHead
    head = FakeHead(question_from_dict(QUESTIONS["urgent"]), [0.9, 0.1])
    log = TrafficLog(tmp_path / "traffic.jsonl")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(
        Decider(RecordingBackend(LOGITS)), "fake", log, heads=[head]))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", log, head
    server.shutdown()


def test_routed_ask_reports_who_answered_each_question(routed_server):
    url, _, head = routed_server
    body = ask(url).json()
    assert body["served_by"]["urgent"] == f"head:{head.family}"
    assert body["served_by"]["intent"] == body["served_by"]["severity"] == "teacher"
    assert body["answers"]["urgent"]["noul"] == pytest.approx(0.9)


def test_the_log_remembers_which_answers_came_from_a_head(routed_server):
    from typed_decisions.traffic import examples
    url, log, head = routed_server
    ask(url)
    sources = {e.question.instructions: e.source for e in examples(log)}
    assert sources["Is this urgent?"] == f"head:{head.family}"
    assert sources["Classify the intent."] == "teacher"


def test_heads_endpoint_lists_what_is_served(routed_server):
    url, _, head = routed_server
    body = httpx.get(f"{url}/v1/heads").json()
    assert [h["family"] for h in body["heads"]] == [head.family]
    assert body["heads"][0]["evaluation"]["promoted"] is True


def test_load_heads_serves_only_promoted_heads(tiny_base, tmp_path):
    from typed_decisions.distill import train_head
    from typed_decisions.routing import load_heads
    from typed_decisions.traffic import Example, family_key
    q = question_from_dict(QUESTIONS["urgent"])
    ex = [Example("r", family_key(q), q, {"ticket": "refund"}, [0.8, 0.2], None)]
    head = train_head(ex, base=tiny_base, device="cpu", log=lambda *_: None)
    head.evaluation = {"promoted": False, "reason": "student Brier worse than teacher"}
    head.save(tmp_path / "heads" / head.family)
    served, skipped = load_heads(tmp_path / "heads", device="cpu")
    assert served == [] and skipped == [(head.family, "student Brier worse than teacher")]


def test_shadow_answers_are_logged_but_never_returned(tmp_path):
    from typed_decisions.traffic import TrafficLog, examples
    from tests.doubles import FakeHead
    shadow = FakeHead(question_from_dict(QUESTIONS["urgent"]), [0.3, 0.7], promoted=False)
    log = TrafficLog(tmp_path / "t.jsonl")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(
        Decider(RecordingBackend(LOGITS)), "fake", log, shadow_heads=[shadow]))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        body = ask(url).json()
        heads = httpx.get(f"{url}/v1/heads").json()
    finally:
        server.shutdown()
    assert "shadow" not in body and body["served_by"]["urgent"] == "teacher"
    assert heads["shadow"] == [{"family": shadow.family, "version": "v1"}]
    urgent = [e for e in examples(log) if e.question.instructions == "Is this urgent?"][0]
    assert urgent.source == "teacher" and urgent.shadow["probabilities"] == [0.3, 0.7]


def test_presets_fill_only_what_was_not_set_explicitly():
    import argparse
    from typed_decisions import server
    captured = {}

    def fake_build(args):
        captured.update(vars(args))
        raise SystemExit(0)

    parser_args = ["--preset", "apple", "--permutations", "2"]
    orig = server.build_client
    server.build_client = fake_build
    try:
        with pytest.raises(SystemExit):
            server.main(parser_args)
    finally:
        server.build_client = orig
    assert captured["backend"] == "mlx" and captured["model"] == "mlx-community/Qwen3.5-9B-4bit"
    assert captured["contextual"] is True
    assert captured["permutations"] == 2            # explicit flag wins over the preset


def test_cli_dispatches_and_reports_unknown_commands(capsys):
    from typed_decisions.cli import main
    main(["help"])
    assert "typed-decisions serve" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="unknown command"):
        main(["frobnicate"])
