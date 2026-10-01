"""The whole Phase 3 loop through the real CLIs, as a user would run it.

    python examples/phase3_loop.py [--workdir DIR]

1. `python -m typed_decisions.server --log` answers the train tickets over HTTP; outcomes are
   POSTed back against each request_id.
2. `python -m typed_decisions.distill train` trains a head from that log and gates it against
   the teacher on held-out outcomes.
3. Half the unseen test tickets go through `--shadow-heads`: the teacher answers, the
   head runs alongside and is logged, outcomes are posted.
4. `python -m typed_decisions.distill shadow --write` gates the head on those real outcomes.
5. The server restarts with `--heads` and answers the OTHER half of the test tickets,
   which neither training nor either gate ever saw.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

import httpx

sys.path.insert(0, os.path.dirname(__file__))
from calibrate import DATASET, ROUTES, stratified_split  # noqa: E402

from typed_decisions import Prediction, report  # noqa: E402
from typed_decisions.traffic import family_key  # noqa: E402
from typed_decisions.types import question_from_dict  # noqa: E402

TEACHER = "Qwen/Qwen3.5-4B"
QUESTION = {"type": "choice", "instructions": "Which team should handle this ticket?",
            "criteria": ROUTES}
PORT = 8765
URL = f"http://127.0.0.1:{PORT}"


def serve(*extra: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "typed_decisions.server", "--model", TEACHER, "--port", str(PORT),
         "--permutations", "4", "--contextual", *extra],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(600):
        try:
            if httpx.get(f"{URL}/healthz", timeout=1).status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        if proc.poll() is not None:
            raise SystemExit(f"server exited:\n{proc.stdout.read()}")
        time.sleep(1)
    proc.kill()
    raise SystemExit("server did not come up")


def ask(text: str) -> tuple[dict, float]:
    start = time.perf_counter()
    r = httpx.post(f"{URL}/v1/ask", timeout=120,
                   json={"state": {"ticket": text}, "questions": {"route": QUESTION}})
    r.raise_for_status()
    return r.json(), (time.perf_counter() - start) * 1e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=tempfile.mkdtemp(prefix="td-phase3-"))
    parser.add_argument("--out", default="phase3_loop.md")
    parser.add_argument("--min-eval", default="30")
    args = parser.parse_args()
    log_path = os.path.join(args.workdir, "traffic.jsonl")
    heads_dir = os.path.join(args.workdir, "heads")
    train, test = stratified_split(DATASET)
    family = family_key(question_from_dict(QUESTION))

    print("1. teacher serves + logs the train tickets", flush=True)
    server = serve("--log", log_path)
    try:
        for text, label in train:
            body, _ = ask(text)
            httpx.post(f"{URL}/v1/outcomes", json={"request_id": body["request_id"],
                                                   "outcomes": {"route": label}}).raise_for_status()
    finally:
        server.terminate()
        server.wait()

    print("2. distil + gate", flush=True)
    subprocess.run([sys.executable, "-m", "typed_decisions.distill", "train", "--log", log_path,
                    "--family", family, "--out", heads_dir, "--holdout", "0.5",
                    "--min-eval", args.min_eval], check=True)
    meta = json.load(open(os.path.join(heads_dir, family, "head.json")))

    # Test tickets: half go through shadow mode (they feed the gate), the other half are
    # only ever used for the final report, so the table is not graded on the gate's data.
    shadow_set, final_set = test[0::2], test[1::2]
    shadow_log = os.path.join(args.workdir, "shadow.jsonl")
    head_dir = os.path.join(heads_dir, family)

    print("3. shadow: teacher serves, the head runs alongside, outcomes arrive", flush=True)
    server = serve("--log", shadow_log, "--shadow-heads", heads_dir)
    try:
        for text, label in shadow_set:
            body, _ = ask(text)
            httpx.post(f"{URL}/v1/outcomes", json={"request_id": body["request_id"],
                                                   "outcomes": {"route": label}}).raise_for_status()
        teacher_run = [ask(text) for text, _ in final_set]   # baseline, same server
    finally:
        server.terminate()
        server.wait()

    print("4. gate on shadow outcomes", flush=True)
    subprocess.run([sys.executable, "-m", "typed_decisions.distill", "shadow", "--log", shadow_log,
                    "--head", head_dir, "--write", "--min-eval", args.min_eval], check=True)
    shadow_meta = json.load(open(os.path.join(head_dir, "head.json")))

    print("5. serve with --heads, ask the final tickets", flush=True)
    server = serve("--heads", heads_dir)
    try:
        heads = httpx.get(f"{URL}/v1/heads").json()["heads"]
        routed_run = [ask(text) for text, _ in final_set]
    finally:
        server.terminate()
        server.wait()

    def summarise(name, runs):
        preds = [Prediction(b["answers"]["route"]["probabilities"], label)
                 for (b, _), (_, label) in zip(runs, final_set)]
        rep = report(preds)
        who = sorted({b.get("served_by", {}).get("route", "teacher").split(":")[0] for b, _ in runs})
        ms = statistics.median(ms for _, ms in runs)
        return f"| {name} | {'/'.join(who)} | {rep.accuracy:.3f} | {rep.brier:.3f} | {ms:.0f} |"

    def verdict(ev):
        return (f"n_eval={ev['n_eval']}, Brier {ev.get('student_brier', float('nan')):.3f} vs "
                f"{ev.get('teacher_brier', float('nan')):.3f}, acc "
                f"{ev.get('student_accuracy', float('nan')):.3f} vs "
                f"{ev.get('teacher_accuracy', float('nan')):.3f} -> "
                f"{'PROMOTED' if ev['promoted'] else 'refused'}: {'; '.join(ev['reasons'])}")

    lines = [
        "# Phase 3 loop, end to end over HTTP",
        "",
        f"Teacher {TEACHER} (4 permutations + contextual). Gate: paired bootstrap, "
        f"`--min-eval {args.min_eval}`, defaults otherwise (95%, Brier margin 0.02, max "
        "accuracy drop 0.02).",
        "",
        f"1. Holdout gate after training on {len(train)} logged tickets: "
        + verdict(meta["evaluation"]),
        f"2. Shadow gate on {len(shadow_set)} unseen tickets: " + verdict(shadow_meta["evaluation"]),
        f"3. Heads served afterwards: {len(heads)}.",
        "",
        f"Final report on {len(final_set)} tickets that neither training nor the gate saw:",
        "",
        "| server | answered by | acc | Brier | median ms / request |",
        "|---|---|---:|---:|---:|",
        summarise("teacher only", teacher_run),
        summarise("with --heads", routed_run),
    ]
    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
