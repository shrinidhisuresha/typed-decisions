"""Public labelled benchmarks for all three primitives, fetched on demand.

The 160 hand-written tickets in examples/calibrate.py only carry Choice labels. These
tasks add real labels for Score and Noul, and a Choice past the 26-letter alphabet:

    tickets_queue     Choice, 10 queues       Tobi-Bueck/customer-support-tickets  (CC BY-NC 4.0)
    tickets_priority  Score, 3 ordered levels  same dataset, `priority` (low/medium/high)
    tickets_incident  Noul                     same dataset, `type == Incident`
    bitext_category   Choice, 11 categories    bitext/Bitext-customer-support-...   (CDLA-Sharing 1.0)
    bitext_intent     Choice, 27 intents       same dataset -- exercises sequence scoring
    bitext_telco_category Choice, 7 categories Bitext telco (CDLA-Sharing 1.0) -- held out from fine-tuning
    banking77         Choice, 77 intents       PolyAI BANKING77 (CC BY 4.0) -- REAL user queries

Real-label held-out domains for the fine-tuning eval (none appear in its training mixture):

    clinc_domain      Choice, 10 domains       CLINC150 test split (CC BY 3.0) -- real queries
    massive_scenario  Choice, 18 scenarios     MASSIVE en-US test (CC BY 4.0) -- voice requests
    goemotions_emotional Noul                  GoEmotions (Apache-2.0) -- Reddit, single-label rows
    mhs_hate          Score, 3 levels          Measuring Hate Speech (CC BY 4.0) -- IRT hate score

Both datasets are synthetic (LLM-generated), so treat results as a controlled benchmark,
not a proxy for real traffic. The data is NOT redistributed here: each dataset's CSV is
downloaded once from Hugging Face into data/raw/ (gitignored), then sampled locally,
balanced per class, with a fixed seed. (The datasets-server rows API rate-limits the
page scan a balanced sample needs, so it is not used.)
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx

from .prompt import declared_options
from .types import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion

RESOLVE = "https://huggingface.co/datasets/{dataset}/resolve/main/{file}"
# Datasets not hosted as a CSV on the Hub: full URLs, concatenated in order.
URLS = {
    "PolyAI/banking77": [
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
    ],
}
FILES = {
    "Tobi-Bueck/customer-support-tickets": "aa_dataset-tickets-multi-lang-5-2-50-version.csv",
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset":
        "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv",
    "bitext/Bitext-telco-llm-chatbot-training-dataset":
        "bitext-telco-llm-chatbot-training-dataset.csv",
}

# Held-out real-label datasets that are not single CSVs: rows come from READERS below.
CLINC = "clinc/oos-eval"
MASSIVE = "mteb/amazon_massive_scenario"
GOEMOTIONS = "google-research-datasets/go_emotions"
MHS = "ucberkeley-dlab/measuring-hate-speech"
CLINC_DOMAINS = ["banking", "credit cards", "kitchen and dining", "home", "auto and commute",
                 "travel", "utility", "work", "small talk", "meta"]
MASSIVE_SCENARIOS = ["alarm", "audio", "calendar", "cooking", "datetime", "email", "general",
                     "iot", "lists", "music", "news", "play", "qa", "recommendation", "social",
                     "takeaway", "transport", "weather"]
GOEMOTIONS_NEUTRAL = 27            # index of "neutral" in the simplified label set
MHS_LEVELS = {"not hateful": 1.0, "somewhat hateful": 2.0, "hateful": 3.0}


def _mhs_level(score: float) -> str:
    # The dataset's own guidance: > 0.5 is approximately hate speech, < -1 counter/supportive.
    return "hateful" if score > 0.5 else "not hateful" if score < -1 else "somewhat hateful"


TICKETS = "Tobi-Bueck/customer-support-tickets"
BITEXT = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
BANKING = "PolyAI/banking77"
TELCO = "bitext/Bitext-telco-llm-chatbot-training-dataset"
TELCO_CATEGORIES = ["BILLING", "COMPLAINTS", "CONSUMPTION", "CONTACT", "PAYMENT", "SERVICES",
                    "SUBSCRIPTION"]
# The 77 categories, as they appear in the dataset (generated from it, not typed).
BANKING77 = [
    "Refund_not_showing_up",
    "activate_my_card",
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_about_to_expire",
    "card_acceptance",
    "card_arrival",
    "card_delivery_estimate",
    "card_linking",
    "card_not_working",
    "card_payment_fee_charged",
    "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate",
    "card_swallowed",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "change_pin",
    "compromised_card",
    "contactless_not_working",
    "country_support",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "direct_debit_payment_not_recognised",
    "disposable_card_limits",
    "edit_personal_details",
    "exchange_charge",
    "exchange_rate",
    "exchange_via_app",
    "extra_charge_on_statement",
    "failed_transfer",
    "fiat_currency_support",
    "get_disposable_virtual_card",
    "get_physical_card",
    "getting_spare_card",
    "getting_virtual_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "order_physical_card",
    "passcode_forgotten",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_top_up",
    "pending_transfer",
    "pin_blocked",
    "receiving_money",
    "request_refund",
    "reverted_card_payment?",
    "supported_cards_and_currencies",
    "terminate_account",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_by_cash_or_cheque",
    "top_up_failed",
    "top_up_limits",
    "top_up_reverted",
    "topping_up_by_card",
    "transaction_charged_twice",
    "transfer_fee_charged",
    "transfer_into_account",
    "transfer_not_received_by_recipient",
    "transfer_timing",
    "unable_to_verify_identity",
    "verify_my_identity",
    "verify_source_of_funds",
    "verify_top_up",
    "virtual_card_not_working",
    "visa_or_mastercard",
    "why_verify_identity",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]

QUEUES = ["Technical Support", "Product Support", "Customer Service", "IT Support",
          "Billing and Payments", "Returns and Exchanges", "Service Outages and Maintenance",
          "Sales and Pre-Sales", "Human Resources", "General Inquiry"]
# The file used has low/medium/high only; very_low/critical exist in other versions.
PRIORITY = {"low": "low", "medium": "medium", "high": "high"}
CATEGORIES = ["ACCOUNT", "CANCEL", "CONTACT", "DELIVERY", "FEEDBACK", "INVOICE", "ORDER",
              "PAYMENT", "REFUND", "SHIPPING", "SUBSCRIPTION"]
INTENTS = ["cancel_order", "change_order", "change_shipping_address", "check_cancellation_fee",
           "check_invoice", "check_payment_methods", "check_refund_policy", "complaint",
           "contact_customer_service", "contact_human_agent", "create_account",
           "delete_account", "delivery_options", "delivery_period", "edit_account",
           "get_invoice", "get_refund", "newsletter_subscription", "payment_issue",
           "place_order", "recover_password", "registration_problems", "review",
           "set_up_shipping_address", "switch_account", "track_order", "track_refund"]


def _ticket_state(row: dict) -> dict:
    return {"subject": row.get("subject") or "", "body": row.get("body") or ""}


def _english(row: dict) -> bool:
    return row.get("language") == "en" and bool(row.get("body"))


@dataclass(frozen=True)
class Task:
    name: str
    dataset: str
    question: Question
    label: Callable[[dict], str | None]       # row -> option text, or None to skip
    state: Callable[[dict], object]
    license: str
    notes: str = ""
    labels: list[str] = field(default_factory=list)

    def __post_init__(self):
        object.__setattr__(self, "labels", declared_options(self.question))


TASKS = {t.name: t for t in [
    Task("tickets_queue", TICKETS,
         ChoiceQuestion("Which support queue should handle this ticket?", QUEUES),
         lambda r: r["queue"] if _english(r) and r.get("queue") in QUEUES else None,
         _ticket_state, "CC BY-NC 4.0"),
    Task("tickets_priority", TICKETS,
         ScoreQuestion("How urgent is this ticket?",
                       {"low": 1.0, "medium": 2.0, "high": 3.0}),
         lambda r: PRIORITY.get(r.get("priority")) if _english(r) else None,
         _ticket_state, "CC BY-NC 4.0",
         "evenly spaced legend (1..3)"),
    Task("tickets_incident", TICKETS,
         NoulQuestion("Does this ticket report an incident: something broken, down or "
                      "not working, as opposed to a request, question or change?"),
         lambda r: (("yes" if r["type"] == "Incident" else "no")
                    if _english(r) and r.get("type") else None),
         _ticket_state, "CC BY-NC 4.0"),
    Task("bitext_category", BITEXT,
         ChoiceQuestion("What is this customer message about?",
                        [c.lower() for c in CATEGORIES]),
         lambda r: r["category"].lower() if r.get("category") in CATEGORIES else None,
         lambda r: r["instruction"], "CDLA-Sharing 1.0"),
    Task("bitext_intent", BITEXT,
         ChoiceQuestion("What does the customer want?", [i.replace("_", " ") for i in INTENTS]),
         lambda r: r["intent"].replace("_", " ") if r.get("intent") in INTENTS else None,
         lambda r: r["instruction"], "CDLA-Sharing 1.0",
         "27 options: scored by option text (sequence scoring)"),
    Task("bitext_telco_category", TELCO,
         ChoiceQuestion("What is this telecom customer's message about?",
                        [c.lower() for c in TELCO_CATEGORIES]),
         lambda r: r["category"].lower() if r.get("category") in TELCO_CATEGORIES else None,
         lambda r: r["instruction"], "CDLA-Sharing 1.0"),
    Task("clinc_domain", CLINC,
         ChoiceQuestion("Which area is this request about?", CLINC_DOMAINS),
         lambda r: r["domain"], lambda r: r["text"], "CC BY 3.0",
         "CLINC150 test split, in-scope queries only; intent mapped to its domain"),
    Task("massive_scenario", MASSIVE,
         ChoiceQuestion("What is this voice-assistant request about?", MASSIVE_SCENARIOS),
         lambda r: r["label_text"] if r["label_text"] in MASSIVE_SCENARIOS else None,
         lambda r: r["text"], "CC BY 4.0"),
    Task("goemotions_emotional", GOEMOTIONS,
         NoulQuestion("Does this comment express an emotion, rather than being neutral?"),
         lambda r: ("no" if r["labels"] == [GOEMOTIONS_NEUTRAL] else "yes")
         if len(r["labels"]) == 1 else None,
         lambda r: r["text"], "Apache-2.0", "single-label rows only"),
    Task("mhs_hate", MHS,
         ScoreQuestion("How hateful is this comment?", MHS_LEVELS),
         lambda r: _mhs_level(r["hate_speech_score"]), lambda r: r["text"], "CC BY 4.0",
         "one row per comment; levels from the IRT hate_speech_score"),
    Task("banking77", BANKING,
         ChoiceQuestion("What does the bank customer want?",
                        [c.replace("_", " ") for c in BANKING77]),
         lambda r: r["category"].replace("_", " ") if r.get("category") in BANKING77 else None,
         lambda r: r["text"], "CC BY 4.0",
         "77 options, real user queries; > 26 so retrieval or option-text scoring applies"),
]}


@dataclass(frozen=True)
class Sample:
    state: object
    label: str


def sample(task: Task, per_class: int, rows, seed: int = 0) -> list[Sample]:
    """Balanced: exactly per_class rows of every label, chosen by a seeded shuffle."""
    rng = random.Random(f"{task.name}:{seed}")
    pool = list(rows)
    rng.shuffle(pool)
    buckets: dict[str, list[Sample]] = {label: [] for label in task.labels}
    for row in pool:
        label = task.label(row)
        if label in buckets and len(buckets[label]) < per_class:
            buckets[label].append(Sample(task.state(row), label))
            if all(len(b) >= per_class for b in buckets.values()):
                break
    short = {k: len(v) for k, v in buckets.items() if len(v) < per_class}
    if short:
        raise RuntimeError(f"{task.name}: not enough rows for every class: {short}")
    out = [s for label in task.labels for s in buckets[label]]
    rng.shuffle(out)
    return out


def _fetch(url: str, path: Path, client: httpx.Client | None) -> None:
    client = client or httpx.Client(timeout=300, follow_redirects=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with client.stream("GET", url) as r:
        r.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in r.iter_bytes():
                fh.write(chunk)
    tmp.rename(path)


def download(dataset: str, cache_dir: str | Path = "data",
             client: httpx.Client | None = None) -> list[Path]:
    """The dataset's CSV file(s), fetched once into <cache_dir>/raw/."""
    folder = Path(cache_dir) / "raw" / dataset.replace("/", "__")
    urls = URLS.get(dataset) or [RESOLVE.format(dataset=dataset, file=FILES[dataset])]
    paths = []
    for url in urls:
        path = folder / url.rsplit("/", 1)[-1]
        if not path.exists():
            _fetch(url, path, client)
        paths.append(path)
    return paths


def read_rows(paths) -> list[dict]:
    csv.field_size_limit(1 << 24)
    rows: list[dict] = []
    for path in [paths] if isinstance(paths, Path) else paths:
        with Path(path).open(newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def _fetch_once(url: str, path: Path, client: httpx.Client | None) -> Path:
    if not path.exists():
        _fetch(url, path, client)
    return path


def _clinc(cache_dir: Path, client) -> list[dict]:
    base = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/"
    folder = cache_dir / "raw" / "clinc__clinc_oos"
    full = json.loads(_fetch_once(base + "data_full.json", folder / "data_full.json", client).read_text())
    domains = json.loads(_fetch_once(base + "domains.json", folder / "domains.json", client).read_text())
    to_domain = {intent: d.replace("_", " ") for d, intents in domains.items() for intent in intents}
    return [{"text": text, "domain": to_domain[intent]} for text, intent in full["test"]]


def _massive(cache_dir: Path, client) -> list[dict]:
    import gzip
    path = _fetch_once(RESOLVE.format(dataset=MASSIVE, file="test/en.json.gz"),
                       cache_dir / "raw" / "mteb__amazon_massive_scenario" / "test-en.json.gz", client)
    with gzip.open(path, "rt") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _parquet(dataset: str, file: str, local: str, cache_dir: Path, client, columns=None) -> list[dict]:
    import pyarrow.parquet as pq   # in the `train` extra
    path = _fetch_once(RESOLVE.format(dataset=dataset, file=file),
                       cache_dir / "raw" / dataset.replace("/", "__") / local, client)
    return pq.read_table(path, columns=columns).to_pylist()


def _goemotions(cache_dir: Path, client) -> list[dict]:
    return _parquet(GOEMOTIONS, "simplified/train-00000-of-00001.parquet", "simplified-train.parquet",
                    cache_dir, client)


def _mhs(cache_dir: Path, client) -> list[dict]:
    rows = _parquet(MHS, "measuring-hate-speech.parquet", "measuring-hate-speech.parquet", cache_dir,
                    client, columns=["comment_id", "hate_speech_score", "text"])
    one: dict = {}
    for r in rows:                 # one row per annotator; the IRT score is per comment
        one.setdefault(r["comment_id"], r)
    return list(one.values())


READERS: dict[str, Callable[[Path, httpx.Client | None], list[dict]]] = {
    CLINC: _clinc, MASSIVE: _massive, GOEMOTIONS: _goemotions, MHS: _mhs,
}


def load(task_name: str, per_class: int, seed: int = 0, cache_dir: str | Path = "data",
         client: httpx.Client | None = None) -> list[Sample]:
    """Cached sample(); the cache key includes everything that changes the sample."""
    task = TASKS[task_name]
    path = Path(cache_dir) / f"{task_name}-{per_class}-{seed}.json"
    if path.exists():
        return [Sample(**s) for s in json.loads(path.read_text())]
    raw = (READERS[task.dataset](Path(cache_dir), client) if task.dataset in READERS
           else read_rows(download(task.dataset, cache_dir, client)))
    rows = sample(task, per_class, raw, seed)
    path.write_text(json.dumps([{"state": s.state, "label": s.label} for s in rows]))
    return rows


def split(samples: list[Sample]) -> tuple[list[Sample], list[Sample]]:
    """Alternate within each label: both halves see every class, in equal numbers."""
    by_label: dict[str, list[Sample]] = {}
    for s in samples:
        by_label.setdefault(s.label, []).append(s)
    train, test = [], []
    for group in by_label.values():
        train.extend(group[0::2])
        test.extend(group[1::2])
    return train, test
