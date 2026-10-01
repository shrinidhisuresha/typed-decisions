import httpx
import pytest

from typed_decisions.benchmarks import TASKS, load, sample, split
from typed_decisions.prompt import declared_options


def fake_hf(rows, calls=None):
    """Serves `rows` as the dataset's CSV."""
    import csv
    import io

    def handler(request):
        if calls is not None:
            calls.append(str(request.url))
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        return httpx.Response(200, content=buf.getvalue().encode())
    return httpx.Client(transport=httpx.MockTransport(handler))


def ticket(i, queue="Technical Support", priority="high", type_="Incident", lang="en"):
    return {"subject": f"s{i}", "body": f"body {i}", "queue": queue, "priority": priority,
            "type": type_, "language": lang}


def test_every_label_is_one_of_its_questions_options():
    for task in TASKS.values():
        assert task.labels == declared_options(task.question)


def test_samples_are_balanced_and_skip_rows_that_do_not_qualify():
    rows = ([ticket(i, type_="Incident") for i in range(300)] +
            [ticket(i, type_="Request") for i in range(300, 600)] +
            [ticket(i, lang="de") for i in range(600, 700)])
    got = sample(TASKS["tickets_incident"], 20, rows)
    assert sorted(s.label for s in got) == ["no"] * 20 + ["yes"] * 20
    assert all(s.state["body"].startswith("body") for s in got)


def test_priority_labels_map_to_the_score_legend():
    rows = [ticket(i, priority=p) for i, p in enumerate(["low", "medium", "high"] * 30)]
    got = sample(TASKS["tickets_priority"], 5, rows)
    assert {s.label for s in got} == {"low", "medium", "high"}


def test_the_sample_is_reproducible_for_a_seed_and_differs_across_seeds():
    rows = [ticket(i, type_=("Incident" if i % 2 else "Change")) for i in range(2000)]
    a = sample(TASKS["tickets_incident"], 10, rows, seed=1)
    b = sample(TASKS["tickets_incident"], 10, rows, seed=1)
    c = sample(TASKS["tickets_incident"], 10, rows, seed=2)
    assert a == b and a != c


def test_an_unfillable_class_is_an_error_not_a_silent_imbalance():
    rows = [ticket(i, type_="Incident") for i in range(300)]
    with pytest.raises(RuntimeError, match="no"):
        sample(TASKS["tickets_incident"], 5, rows)


def test_load_caches_so_a_second_call_makes_no_requests(tmp_path):
    rows = [ticket(i, type_=("Incident" if i % 2 else "Change")) for i in range(400)]
    calls = []
    first = load("tickets_incident", 5, cache_dir=tmp_path, client=fake_hf(rows, calls))
    n = len(calls)
    again = load("tickets_incident", 5, cache_dir=tmp_path, client=fake_hf(rows, calls))
    assert again == first and len(calls) == n


def test_split_halves_every_class():
    from typed_decisions.benchmarks import Sample
    samples = [Sample(i, lab) for i, lab in enumerate(["a"] * 10 + ["b"] * 10)]
    train, test = split(samples)
    assert sorted(s.label for s in train) == sorted(s.label for s in test) == ["a"] * 5 + ["b"] * 5
