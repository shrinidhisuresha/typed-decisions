"""One state, three typed questions, zero generated tokens.

    python examples/demo.py [model]
"""

import sys
import json

from typed_decisions import Decider, ChoiceQuestion, NoulQuestion, ScoreQuestion
from typed_decisions.backend_impls.transformers import TransformersBackend

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-0.8B"

STATE = {
    "ticket": "I was charged twice for my subscription this month. Please refund the duplicate.",
    "customer_tier": "enterprise",
    "account_age_days": 412,
}

QUESTIONS = {
    "route": ChoiceQuestion(
        "Which team should handle this ticket?",
        ["billing", "technical support", "sales", "account management"],
    ),
    "needs_human": NoulQuestion("Does this require a human agent rather than automation?"),
    "involves_money": NoulQuestion("Does this ticket involve a payment or charge?"),
    "severity": ScoreQuestion(
        "Rate the severity of this ticket.",
        {"trivial": 1.0, "minor": 2.0, "moderate": 3.0, "serious": 4.0, "critical": 5.0},
    ),
}


def main() -> None:
    client = Decider(TransformersBackend.from_pretrained(MODEL))
    result = client.ask(STATE, QUESTIONS)

    print(json.dumps(result.to_dict(), indent=2))
    print()
    print(f"questions asked : {len(QUESTIONS)}")
    print(f"input tokens    : {result.usage.input_tokens}")
    print(f"output tokens   : {result.usage.output_tokens}  <- nothing was generated")
    print()
    print("Reminder: confidence here is UNCALIBRATED (Phase 0). Do not gate on it.")


if __name__ == "__main__":
    main()
