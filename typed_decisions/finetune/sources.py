"""Training sources for the mixture: public datasets, none of them used by eval.py.

Each source becomes rows of one or more question SHAPES -- Choice, Noul, Score -- so the
scoring rule is optimised over every shape the evaluation asks. Held-out evaluation
datasets (bitext, the tickets set) must never appear here.

Licences (check before publishing any adapter trained on these):
  banking77   CC BY 4.0
  trec        public research data (Li & Roth)
  ag_news     academic, non-commercial use     -> experiment only
  enron_spam  public (released by FERC)
  subj, sst   no explicit licence (Pang & Lee; Stanford) -> experiment only
  amazon      Amazon reviews (multilingual corpus licence, non-commercial) -> experiment only
  civil       Civil Comments / Jigsaw, CC0
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..benchmarks import BANKING, BANKING77, RESOLVE, _fetch, read_rows
from ..benchmarks import download as download_csv

MAX_CHARS = 1500


@dataclass(frozen=True)
class Source:
    name: str
    dataset: str
    file: str                         # jsonl in the HF repo; "" for the CSV-backed BANKING77
    text: Callable[[dict], str]
    label: Callable[[dict], str | None]
    labels: list[str]
    shape: str                        # "choice" | "noul" | "score" | "soft_noul"
    per_class: int
    license: str
    noul_questions: tuple[str, ...] = ()        # for shape "noul": phrasings whose "yes" is labels[0]
    negated_questions: tuple[str, ...] = ()     # phrasings whose "yes" is labels[1]
    score_questions: tuple[str, ...] = ()       # for shape "score": "how much of labels[-1]"
    collapse: dict[str, str] = field(default_factory=dict)   # optional coarser legend
    choice_questions: tuple[str, ...] = ()      # for shape "choice"; default: data.CHOICE_INSTRUCTIONS
    about_questions: tuple[str, ...] = ()       # Noul "is it {x}?" over a choice source's labels
    # For shape "soft_noul": P(yes) for a row, e.g. the fraction of annotators who said yes.
    # Training on it teaches hedging where people genuinely disagree.
    soft: Callable[[dict], float] | None = None


def _jsonl(dataset: str, file: str, cache_dir: str | Path) -> list[dict]:
    path = Path(cache_dir) / "raw" / dataset.replace("/", "__") / file
    if not path.exists():
        _fetch(RESOLVE.format(dataset=dataset, file=file), path, None)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _parquet(dataset: str, file: str, cache_dir: str | Path) -> list[dict]:
    import pyarrow.parquet as pq   # in the `train` extra

    path = Path(cache_dir) / "raw" / dataset.replace("/", "__") / Path(file).name
    if not path.exists():
        _fetch(RESOLVE.format(dataset=dataset, file=file), path, None)
    return pq.read_table(path).to_pylist()


def rows_for(source: Source, cache_dir: str | Path = "data") -> list[dict]:
    if not source.file:
        return read_rows(download_csv(source.dataset, cache_dir))
    if source.file.endswith(".parquet"):
        return _parquet(source.dataset, source.file, cache_dir)
    return _jsonl(source.dataset, source.file, cache_dir)


SST5 = ["very negative", "negative", "neutral", "positive", "very positive"]
STARS = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
TOXICITY = ["not toxic", "somewhat toxic", "toxic", "very toxic"]


def _toxicity_level(t: float) -> str:
    return TOXICITY[0 if t < 0.1 else 1 if t < 0.4 else 2 if t < 0.7 else 3]


TREC = ["abbreviation", "entity", "description", "person", "location", "number"]

SOURCES = {s.name: s for s in [
    Source("banking77", BANKING, "", lambda r: r["text"],
           lambda r: r["category"].replace("_", " ") if r.get("category") in BANKING77 else None,
           [c.replace("_", " ") for c in BANKING77], "choice", 20, "CC BY 4.0"),
    Source("ag_news", "SetFit/ag_news", "train.jsonl", lambda r: r["text"],
           lambda r: r["label_text"].lower(), ["world", "sports", "business", "sci/tech"],
           "choice", 150, "academic non-commercial",
           choice_questions=("What is this news article about?",
                             "Which section of the paper does this story belong in?"),
           about_questions=("Is this article about {x}?", "Does this story belong in {x}?")),
    Source("trec", "SetFit/TREC-QC", "train.jsonl", lambda r: r["text"],
           lambda r: TREC[r["label_coarse"]], TREC, "choice", 100, "public research data",
           choice_questions=("What type of answer does this question expect?",
                             "What kind of thing is this question asking for?"),
           about_questions=("Does this question ask for a {x}?",)),
    Source("enron_spam", "SetFit/enron_spam", "train.jsonl",
           lambda r: f"Subject: {r.get('subject') or ''}\n{r.get('message') or ''}",
           lambda r: r["label_text"], ["spam", "ham"], "noul", 300, "public",
           noul_questions=("Is this email spam?", "Is this unsolicited bulk or scam email?"),
           negated_questions=("Is this a legitimate, non-spam email?",)),
    Source("subj", "SetFit/subj", "train.jsonl", lambda r: r["text"],
           lambda r: r["label_text"], ["subjective", "objective"], "noul", 300, "no licence",
           noul_questions=("Is this sentence a subjective opinion?",
                           "Does this express a personal view rather than a fact?"),
           negated_questions=("Is this an objective, factual statement?",)),
    Source("sst2", "SetFit/sst2", "train.jsonl", lambda r: r["text"],
           lambda r: r["label_text"], ["positive", "negative"], "noul", 300, "no licence",
           noul_questions=("Is this review positive?", "Did the reviewer like it?"),
           negated_questions=("Is this review negative?",)),
    Source("sst5", "SetFit/sst5", "train.jsonl", lambda r: r["text"],
           lambda r: r["label_text"], SST5, "score", 150, "no licence",
           score_questions=("How positive is this review?", "Rate the sentiment of this review."),
           collapse={"very negative": "negative", "negative": "negative", "neutral": "neutral",
                     "positive": "positive", "very positive": "positive"}),
    Source("amazon", "SetFit/amazon_reviews_multi_en", "validation.jsonl", lambda r: r["text"],
           lambda r: STARS[int(r["label"])], STARS, "score", 150, "non-commercial",
           score_questions=("How satisfied was this customer?",
                            "How many stars would this review give?"),
           collapse={"1 star": "dissatisfied", "2 stars": "dissatisfied", "3 stars": "mixed",
                     "4 stars": "satisfied", "5 stars": "satisfied"}),
    Source("civil", "google/civil_comments", "data/validation-00000-of-00001.parquet",
           lambda r: r["text"], lambda r: _toxicity_level(r["toxicity"]), TOXICITY, "score",
           150, "CC0",
           score_questions=("How toxic is this comment?", "How offensive is this comment?")),
    Source("civil_soft", "google/civil_comments", "data/validation-00000-of-00001.parquet",
           lambda r: r["text"], lambda r: f"decile {min(int(r['toxicity'] * 10), 9)}",
           [f"decile {i}" for i in range(10)], "soft_noul", 60, "CC0",
           noul_questions=("Is this comment toxic?", "Would readers find this comment offensive?"),
           soft=lambda r: float(r["toxicity"])),
]}

MIXTURES = {
    "banking77": ["banking77"],
    "mix1": ["banking77", "ag_news", "trec", "enron_spam", "subj", "sst2", "sst5"],
    "mix2": ["banking77", "ag_news", "trec", "enron_spam", "subj", "sst2", "sst5", "amazon",
             "civil", "civil_soft"],
}


@dataclass(frozen=True)
class Labelled:
    state: str
    label: str
    soft: float | None = None


def balanced(source: Source, rows: list[dict], labels: list[str], per_class: int,
             relabel: Callable[[str], str] = lambda x: x, seed: int = 0) -> list[Labelled]:
    """per_class rows of every label in `labels`, or as many as exist."""
    rng = random.Random(f"{source.name}:{seed}:{len(labels)}")
    pool = list(rows)
    rng.shuffle(pool)
    buckets: dict[str, list[Labelled]] = {l: [] for l in labels}
    for row in pool:
        raw = source.label(row)
        label = relabel(raw) if raw is not None else None
        text = (source.text(row) or "").strip()
        if label in buckets and text and len(buckets[label]) < per_class:
            buckets[label].append(Labelled(text[:MAX_CHARS], label,
                                           source.soft(row) if source.soft else None))
    return [s for b in buckets.values() for s in b]
