"""Integration: a real instruct model must get easy decisions right.

Marked slow -- downloads weights. Run with: pytest -m slow
Point it at a bigger rung with TD_E2E_MODEL=Qwen/Qwen3.5-4B TD_E2E_DEVICE=mps.
"""

import os

import pytest

torch = pytest.importorskip("torch")

from typed_decisions.client import Decider
from typed_decisions.backend_impls.transformers import TransformersBackend
from typed_decisions.types import ChoiceQuestion, NoulQuestion, ScoreQuestion

MODEL = os.environ.get("TD_E2E_MODEL", "Qwen/Qwen3.5-0.8B")
DEVICE = os.environ.get("TD_E2E_DEVICE", "cpu")

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def client():
    return Decider(TransformersBackend.from_pretrained(
        MODEL, device=DEVICE, dtype=torch.float32 if DEVICE == "cpu" else torch.bfloat16
    ))


STATE = {
    "ticket": "I was charged twice for my subscription this month. Refund the duplicate.",
    "customer_tier": "enterprise",
}


def test_routes_a_billing_ticket_to_billing(client):
    result = client.ask(STATE, {
        "intent": ChoiceQuestion(
            "Which team should handle this ticket?",
            ["billing", "technical support", "sales"],
        )
    })
    assert result.answers["intent"].choice == "billing"


def test_noul_separates_a_true_claim_from_a_false_one(client):
    answers = client.ask(STATE, {
        "money": NoulQuestion("Does the ticket involve a payment or charge?"),
        "outage": NoulQuestion("Does the ticket report a service outage?"),
    }).answers
    assert answers["money"].noul > answers["outage"].noul


@pytest.mark.xfail(
    MODEL.endswith(("-0.8B", "-2B")),
    reason="below 4B the model cannot rank severity: measured 0.8B raw 1.19 vs 1.37, "
    "debiased 2.98 vs 3.01; 2B 2.09 vs 2.03; 4B 4.60 vs 2.06. Capability, not code.",
    strict=True,
)
def test_score_orders_a_severe_state_above_a_mild_one(client):
    question = {"sev": ScoreQuestion(
        "Rate the severity of this ticket.",
        {"trivial": 1.0, "moderate": 3.0, "critical": 5.0},
    )}
    severe = client.ask(
        {"ticket": "Production is down for all users and we are losing revenue."}, question
    ).answers["sev"].score
    mild = client.ask(
        {"ticket": "The logo on the settings page is slightly misaligned."}, question
    ).answers["sev"].score
    assert severe > mild


def test_no_output_tokens_are_billed_end_to_end(client):
    result = client.ask(STATE, {"intent": ChoiceQuestion("Route it.", ["billing", "sales"])})
    assert result.usage.output_tokens == 0
