"""Re-score saved predictions with estimators that hold up at small n.

    python -m typed_decisions.finetune.rescore --model-tag base-Qwen3.5-4B-4bit \\
        --tuned mix2=mix2-4b-r16 --tuned penalty=mix2cps-4b-r16-s0,mix2cps-4b-r16-s1,mix2cps-4b-r16-s2 \\
        --out docs/rescore_qwen3.5-4b.md

eval.py's first gate was underpowered: at n=75-220 a perfectly calibrated model still scores
10-bin ECE of about 0.1, and the paired Brier CI is wider than the 0.02 margin it tests. This
module needs no new inference. It reads data/finetune_dists/ and changes how the predictions are scored:

- NLL and Brier are primary (RPS as well for Score tasks). ECE uses equal-mass bins and is shown
  next to its own noise floor: the ECE a perfectly calibrated model with these predictions would
  score at this n.
- The temperature is cross-fitted (5 folds over all rows) instead of fitted on a 50/50 split, so
  every row is scored and n doubles. The fitted T gets a bootstrap CI.
- Calibration intercept and slope (top-label logistic recalibration) separate a shift from
  over- or under-confidence: perfect is (0, 1), and a slope below 1 is overconfident.
- The gate returns PASS, FAIL or INCONCLUSIVE. INCONCLUSIVE means the CI is wider than the margin
  and n cannot decide. One pooled endpoint (mean Brier difference over tasks, stratified bootstrap)
  is the primary claim; per-task verdicts are secondary.
- Several adapters (seeds) per config report a per-seed verdict count and a two-level bootstrap
  (seeds, then rows).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..benchmarks import TASKS
from ..types import ScoreQuestion

TASK_NAMES = ["bitext_category", "bitext_telco_category", "tickets_incident", "tickets_priority"]
FLOOR = 1e-6


@dataclass
class Scored:
    probs: np.ndarray        # (n, K) over task.labels
    labels: np.ndarray       # (n,)


def load(dists: Path, task: str, config: str, tag: str) -> Scored:
    """All rows of a cached prediction file: its fitting half and its test half, pooled."""
    data = json.loads((dists / f"{task}-{config}-{tag}.json").read_text())
    options = TASKS[task].labels
    rows = data["train"] + data["test"]
    probs = np.array([[r["probabilities"][o] for o in options] for r in rows], dtype=float)
    labels = np.array([options.index(r["label"]) for r in rows])
    return Scored(probs, labels)


# -- metrics, per row ------------------------------------------------------------------

def nll_rows(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(y)), y], FLOOR, 1.0))


def brier_rows(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    onehot = np.eye(p.shape[1])[y]
    return ((p - onehot) ** 2).sum(axis=1)


def rps_rows(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ranked probability score; columns must be in the legend's natural order."""
    onehot = np.eye(p.shape[1])[y]
    return ((p.cumsum(1) - onehot.cumsum(1)) ** 2).sum(axis=1) / max(p.shape[1] - 1, 1)


def correct_rows(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (p.argmax(1) == y).astype(float)


def ece_equal_mass(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf, correct = p.max(1), correct_rows(p, y)
    order = np.argsort(conf)
    total = 0.0
    for chunk in np.array_split(order, min(bins, len(order))):
        if len(chunk):
            total += len(chunk) * abs(conf[chunk].mean() - correct[chunk].mean())
    return total / len(conf)


def ece_floor(p: np.ndarray, bins: int = 10, draws: int = 200, seed: int = 0) -> float:
    """Mean ECE of a perfectly calibrated model with these exact predictions at this n:
    draw labels from the predicted distributions and score them. ECE at or below this is noise."""
    rng = np.random.default_rng(seed)
    cum = p.cumsum(1)
    vals = []
    for _ in range(draws):
        u = rng.random((len(p), 1))
        y = np.minimum((u > cum).sum(1), p.shape[1] - 1)
        vals.append(ece_equal_mass(p, y, bins))
    return float(np.mean(vals))


# -- temperature ------------------------------------------------------------------------

def scale(p: np.ndarray, t: float) -> np.ndarray:
    logp = np.log(np.clip(p, FLOOR, 1.0)) / t
    logp -= logp.max(1, keepdims=True)
    q = np.exp(logp)
    return q / q.sum(1, keepdims=True)


def fit_t(p: np.ndarray, y: np.ndarray, lo: float = 0.05, hi: float = 20.0) -> float:
    """Golden-section search over log T on NLL (unimodal in T)."""
    a, b = np.log(lo), np.log(hi)
    inv = (np.sqrt(5) - 1) / 2
    f = lambda lt: nll_rows(scale(p, np.exp(lt)), y).mean()
    c, d = b - inv * (b - a), a + inv * (b - a)
    fc, fd = f(c), f(d)
    while b - a > 1e-4:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv * (b - a)
            fd = f(d)
    return float(np.exp((a + b) / 2))


def folds(y: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    """Fold id per row, stratified by label."""
    rng = np.random.default_rng(seed)
    out = np.empty(len(y), dtype=int)
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        out[idx] = np.arange(len(idx)) % k
    return out


def cross_fit(p: np.ndarray, y: np.ndarray, k: int = 5) -> tuple[np.ndarray, list[float]]:
    """Each row scaled by a T fitted on the other folds; returns (scaled probs, per-fold T)."""
    f = folds(y, k)
    out, ts = np.empty_like(p), []
    for i in range(k):
        held = f == i
        t = fit_t(p[~held], y[~held])
        out[held] = scale(p[held], t)
        ts.append(t)
    return out, ts


def t_interval(p: np.ndarray, y: np.ndarray, resamples: int = 200, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    ts = [fit_t(p[i], y[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(resamples))]
    return float(np.percentile(ts, 2.5)), float(np.percentile(ts, 97.5))


def calibration_line(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(intercept, slope) of logistic regression of top-label correctness on logit(confidence).
    Perfect calibration is (0, 1); a slope < 1 is overconfident and a slope > 1 underconfident."""
    conf = np.clip(p.max(1), 1e-4, 1 - 1e-4)
    x = np.log(conf / (1 - conf))
    target = correct_rows(p, y)
    X = np.column_stack([np.ones_like(x), x])
    prior, ridge = np.array([0.0, 1.0]), 1.0     # weak pull toward perfect calibration, ~1 row

    def objective(w):
        z = X @ w
        return float((np.logaddexp(0, z) - target * z).sum() + 0.5 * ridge * ((w - prior) ** 2).sum())

    # Damped Newton: undamped steps diverge when many confidences sit at the clip
    # (e.g. 1 - 1e-4), which a binary head trained with a proper score produces often.
    w = prior.copy()
    for _ in range(100):
        mu = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        grad = X.T @ (target - mu) - ridge * (w - prior)
        hess = X.T @ (X * (mu * (1 - mu))[:, None]) + ridge * np.eye(2)
        step = np.linalg.solve(hess, grad)
        size, current = 1.0, objective(w)
        while objective(w + size * step) > current and size > 1e-6:
            size /= 2
        w = w + size * step
        if np.abs(size * step).max() < 1e-8:
            break
    return float(w[0]), float(w[1])


# -- the gate ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Policy:
    brier_margin: float = 0.02
    max_accuracy_drop: float = 0.02
    resamples: int = 2000
    seed: int = 0


def verdict(brier_ci: tuple[float, float], acc_ci: tuple[float, float], policy: Policy) -> str:
    """PASS: non-inferior on both. FAIL: clearly worse on either. Otherwise INCONCLUSIVE."""
    if brier_ci[1] <= policy.brier_margin and acc_ci[0] >= -policy.max_accuracy_drop:
        return "PASS"
    if brier_ci[0] > policy.brier_margin or acc_ci[1] < -policy.max_accuracy_drop:
        return "FAIL"
    return "INCONCLUSIVE"


def paired_ci(diffs: np.ndarray, policy: Policy) -> tuple[float, float]:
    rng = np.random.default_rng(policy.seed)
    means = diffs[rng.integers(0, len(diffs), (policy.resamples, len(diffs)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def pooled_ci(per_task: list[np.ndarray], policy: Policy, seeds: list[list[np.ndarray]] | None = None
              ) -> tuple[float, float, float]:
    """Mean over tasks of the mean paired difference. With `seeds` (one per-task list per seed),
    resample seeds first, then rows within each task: the two-level bootstrap."""
    rng = np.random.default_rng(policy.seed)
    groups = seeds if seeds else [per_task]
    point = float(np.mean([np.mean([d.mean() for d in g]) for g in groups]))
    stats = []
    for _ in range(policy.resamples):
        picked = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        stats.append(np.mean([np.mean([d[rng.integers(0, len(d), len(d))].mean() for d in g])
                              for g in picked]))
    return point, float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


# -- report -----------------------------------------------------------------------------

def metrics(task: str, p: np.ndarray, y: np.ndarray) -> dict:
    out = {"acc": correct_rows(p, y).mean(), "nll": nll_rows(p, y).mean(),
           "brier": brier_rows(p, y).mean(), "ece": ece_equal_mass(p, y)}
    if isinstance(TASKS[task].question, ScoreQuestion):
        out["rps"] = rps_rows(p, y).mean()
    out["intercept"], out["slope"] = calibration_line(p, y)
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-tag", required=True, help="e.g. base-Qwen3.5-4B-4bit")
    parser.add_argument("--tuned", action="append", required=True,
                        help="name=adapter_tag[,adapter_tag...]; several tags are seeds")
    parser.add_argument("--tasks", default=",".join(TASK_NAMES))
    parser.add_argument("--dists", default="data/finetune_dists")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    dists, policy = Path(args.dists), Policy()
    tasks = args.tasks.split(",")
    configs = {}
    for spec in args.tuned:
        name, tags = spec.split("=", 1)
        configs[name] = tags.split(",")

    lines = [f"# Re-scored: {args.model_tag}", "",
             "Generated by `typed_decisions.finetune.rescore` from saved predictions. No new inference. "
             "Every row is scored (fitting half and test half pooled). Post-hoc and raw use a "
             "5-fold cross-fitted T; tuned is scored at T=1. ECE uses 10 equal-mass bins; "
             "\"floor\" is the ECE a perfectly calibrated model with the same predictions would "
             "score at this n. Slope < 1 means overconfident.", ""]

    post_cache: dict[str, np.ndarray] = {}
    pooled_inputs: dict[str, dict] = {}
    for task in tasks:
        raw = load(dists, task, "raw", args.model_tag)
        post = load(dists, task, "posthoc", args.model_tag)
        y = post.labels
        post_cf, post_ts = cross_fit(post.probs, y)
        raw_cf, _ = cross_fit(raw.probs, raw.labels)
        post_cache[task] = post_cf
        is_score = isinstance(TASKS[task].question, ScoreQuestion)
        head = "| config | acc | NLL | Brier |" + (" RPS |" if is_score else "") + \
               " ECE (floor) | intercept / slope | T (95% CI) |"
        lines += [f"## {task} (n={len(y)})", "", head,
                  "|---" * (head.count("|") - 1) + "|"]

        def row(name, p, yy, t_note):
            m = metrics(task, p, yy)
            rps = f" {m['rps']:.3f} |" if is_score else ""
            return (f"| {name} | {m['acc']:.3f} | {m['nll']:.3f} | {m['brier']:.3f} |{rps} "
                    f"{m['ece']:.3f} ({ece_floor(p):.3f}) | {m['intercept']:+.2f} / {m['slope']:.2f} | "
                    f"{t_note} |")

        lines.append(row("raw (cross-fit T)", raw_cf, raw.labels, "—"))
        lo, hi = t_interval(post.probs, y)
        lines.append(row("post-hoc (cross-fit T)", post_cf, y,
                         f"{np.median(post_ts):.2f} ({lo:.2f}–{hi:.2f})"))
        for name, tags in configs.items():
            for tag in tags:
                tuned = load(dists, task, "tuned", tag)
                assert np.array_equal(tuned.labels, y), "row order differs from post-hoc"
                lo, hi = t_interval(tuned.probs, y)
                label = name if len(tags) == 1 else f"{name} [{tag.rsplit('-', 1)[-1]}]"
                lines.append(row(f"{label} (T=1)", tuned.probs, y,
                                 f"{fit_t(tuned.probs, y):.2f} ({lo:.2f}–{hi:.2f})"))
                b = brier_rows(tuned.probs, y) - brier_rows(post_cf, y)
                a = correct_rows(tuned.probs, y) - correct_rows(post_cf, y)
                pooled_inputs.setdefault(name, {}).setdefault(tag, {})[task] = (b, a)
        lines.append("")

    lines += ["## Gate: tuned at T=1 vs post-hoc at cross-fitted T", "",
              f"Margins: Brier +{policy.brier_margin}, accuracy -{policy.max_accuracy_drop}. "
              "PASS = non-inferior on both; FAIL = CI excludes the margin on either; "
              "INCONCLUSIVE = the CI straddles it (n too small to decide). "
              "**The pooled row is the primary claim**; per-task rows are secondary.", "",
              "| config | task | Brier diff (95% CI) | accuracy diff (95% CI) | verdict |",
              "|---|---|---|---|---|"]
    for name, by_tag in pooled_inputs.items():
        tags = list(by_tag)
        for task in tasks:
            verdicts, bs, as_ = [], [], []
            for tag in tags:
                b, a = by_tag[tag][task]
                bci, aci = paired_ci(b, policy), paired_ci(a, policy)
                verdicts.append(verdict(bci, aci, policy))
                bs.append((b.mean(), *bci))
                as_.append((a.mean(), *aci))
            if len(tags) == 1:
                (bm, bl, bh), (am, al, ah) = bs[0], as_[0]
                lines.append(f"| {name} | {task} | {bm:+.3f} ({bl:+.3f}, {bh:+.3f}) | "
                             f"{am:+.3f} ({al:+.3f}, {ah:+.3f}) | {verdicts[0]} |")
            else:
                counts = {v: verdicts.count(v) for v in ("PASS", "INCONCLUSIVE", "FAIL")}
                lines.append(f"| {name} | {task} | mean {np.mean([x[0] for x in bs]):+.3f} | "
                             f"mean {np.mean([x[0] for x in as_]):+.3f} | "
                             + ", ".join(f"{k} {v}/{len(tags)}" for k, v in counts.items() if v) + " |")
        seed_groups_b = [[by_tag[t][task][0] for task in tasks] for t in tags]
        seed_groups_a = [[by_tag[t][task][1] for task in tasks] for t in tags]
        bm, bl, bh = pooled_ci(seed_groups_b[0], policy, seed_groups_b if len(tags) > 1 else None)
        am, al, ah = pooled_ci(seed_groups_a[0], policy, seed_groups_a if len(tags) > 1 else None)
        lines.append(f"| **{name}** | **pooled ({len(tasks)} tasks"
                     + (f", {len(tags)} seeds" if len(tags) > 1 else "") + ")** | "
                     f"**{bm:+.3f} ({bl:+.3f}, {bh:+.3f})** | **{am:+.3f} ({al:+.3f}, {ah:+.3f})** | "
                     f"**{verdict((bl, bh), (al, ah), policy)}** |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
