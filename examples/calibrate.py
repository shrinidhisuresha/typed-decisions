"""Phase 1 in anger: measure calibration, fit a temperature, measure again.

Fits on a train split and reports on a held-out test split -- fitting and reporting
on the same examples would flatter the result.

    python examples/calibrate.py [model]
"""

import sys

from typed_decisions import Calibration, Decider, Prediction, diagnose_temperature, report
from typed_decisions.backend_impls.transformers import TransformersBackend
from typed_decisions.types import ChoiceQuestion

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-0.8B"

ROUTES = ["billing", "technical support", "sales", "account management"]
QUESTION = {"route": ChoiceQuestion("Which team should handle this ticket?", ROUTES)}

DATASET = [
    ('I was charged twice for my subscription this month.', 'billing'),
    ('My invoice shows the wrong VAT rate.', 'billing'),
    ('Please refund the duplicate payment on my card.', 'billing'),
    ('Why did my monthly bill go up by $40?', 'billing'),
    ('The receipt for order 8812 never arrived.', 'billing'),
    ('My credit card was declined but I was still charged.', 'billing'),
    ('There is a late fee on my account that I do not understand.', 'billing'),
    ("I need a copy of last quarter's invoices for my accountant.", 'billing'),
    ('The invoice total does not match the amount deducted.', 'billing'),
    ('I was billed after I cancelled last month.', 'billing'),
    ('Can you update the card on file, the old one expired.', 'billing'),
    ('My payment failed three times but the money left my account.', 'billing'),
    ('I need the tax ID added to our invoices.', 'billing'),
    ('Why is there a proration charge on this bill?', 'billing'),
    ('The refund from two weeks ago has not appeared.', 'billing'),
    ('You charged me in USD instead of EUR.', 'billing'),
    ("I need an itemised breakdown of this month's charges.", 'billing'),
    ('The discount code was not applied to my invoice.', 'billing'),
    ('I was charged for seats we removed last cycle.', 'billing'),
    ('Please send the invoice to our accounts payable address.', 'billing'),
    ('There is a $9 charge I do not recognise.', 'billing'),
    ('My annual renewal charged before the renewal date.', 'billing'),
    ('Can you void invoice INV-3391 and reissue it?', 'billing'),
    ('The payment portal says my card is invalid but it works elsewhere.', 'billing'),
    ('I need a credit note for the overcharge.', 'billing'),
    ('Direct debit was taken twice this month.', 'billing'),
    ('The currency conversion fee seems wrong.', 'billing'),
    ('I paid by bank transfer but the account still shows unpaid.', 'billing'),
    ('Please stop the automatic payments on my account.', 'billing'),
    ('The invoice is missing our purchase order number.', 'billing'),
    ('I was charged the full rate despite the non-profit discount.', 'billing'),
    ('My final bill after cancellation looks too high.', 'billing'),
    ('Can you explain the overage charges on this statement?', 'billing'),
    ('The receipt shows a different email than mine.', 'billing'),
    ('I need invoices reissued with our new company name.', 'billing'),
    ('There are two charges on the same date for the same amount.', 'billing'),
    ('The refund was issued to a closed card.', 'billing'),
    ('Why does my statement show a charge from last year?', 'billing'),
    ('I need a W-9 form for our records.', 'billing'),
    ('The subscription fee was deducted despite my trial being active.', 'billing'),
    ('The app crashes every time I open the reports tab.', 'technical support'),
    ('I am getting a 500 error when uploading a CSV.', 'technical support'),
    ('Two-factor codes are never delivered to my phone.', 'technical support'),
    ('The API returns malformed JSON on the /users endpoint.', 'technical support'),
    ('Dark mode does not persist after I reload the page.', 'technical support'),
    ('Sync has been stuck at 40% for three hours.', 'technical support'),
    ('Webhooks stopped firing after the last deploy.', 'technical support'),
    ('I cannot log in, the page just spins forever.', 'technical support'),
    ('Search returns no results even for exact matches.', 'technical support'),
    ('The dashboard shows stale data from yesterday.', 'technical support'),
    ('Exporting to PDF produces a corrupted file.', 'technical support'),
    ('The mobile app force closes on startup.', 'technical support'),
    ('Our integration is timing out after 30 seconds.', 'technical support'),
    ('Rate limiting kicks in far below the documented threshold.', 'technical support'),
    ('Images fail to upload with a network error.', 'technical support'),
    ('The date picker shows the wrong timezone.', 'technical support'),
    ('Notifications are duplicated five times each.', 'technical support'),
    ('The CSV import drops every row with an accent.', 'technical support'),
    ('The password reset email never arrives.', 'technical support'),
    ('Charts render blank in Safari but work in Chrome.', 'technical support'),
    ('The websocket connection drops every few minutes.', 'technical support'),
    ('Filtering by date range returns incorrect totals.', 'technical support'),
    ('The undo button does not restore deleted items.', 'technical support'),
    ('API keys stopped working after the maintenance window.', 'technical support'),
    ('Pagination skips the last page of results.', 'technical support'),
    ('The editor loses my changes when I switch tabs.', 'technical support'),
    ('Bulk delete only removes the first 100 records.', 'technical support'),
    ('The page takes 40 seconds to load for large accounts.', 'technical support'),
    ('Attachments download as zero byte files.', 'technical support'),
    ('The mobile layout overlaps on small screens.', 'technical support'),
    ('Scheduled reports stopped sending last Tuesday.', 'technical support'),
    ('The API docs example returns a 404.', 'technical support'),
    ('Sorting by name puts uppercase entries first.', 'technical support'),
    ('The app shows a blank screen after the latest update.', 'technical support'),
    ('Our staging webhook receives production events.', 'technical support'),
    ('Copy to clipboard does nothing in Firefox.', 'technical support'),
    ('The audit log is missing entries from last week.', 'technical support'),
    ('Autosave fires every keystroke and slows the editor.', 'technical support'),
    ('The import wizard rejects valid email addresses.', 'technical support'),
    ('Session expires after two minutes instead of an hour.', 'technical support'),
    ('What does your enterprise tier cost for 200 seats?', 'sales'),
    ('We would like a demo before our procurement review.', 'sales'),
    ('Do you offer a discount for non-profits?', 'sales'),
    ('Can you send a quote for a three year contract?', 'sales'),
    ('Is there a free trial for the analytics add-on?', 'sales'),
    ('We are evaluating you against a competitor, can we talk?', 'sales'),
    ('What is included in the professional plan?', 'sales'),
    ('I need pricing for adding a second workspace.', 'sales'),
    ('Do you have a partner or reseller programme?', 'sales'),
    ('Can we get a security questionnaire completed before purchase?', 'sales'),
    ('What is the price difference between team and business?', 'sales'),
    ('We need a proposal for our board by Friday.', 'sales'),
    ('Do you offer volume discounts above 500 users?', 'sales'),
    ('Is there an educational discount for universities?', 'sales'),
    ('Can we pilot the product with 10 users for a month?', 'sales'),
    ('What does onboarding support cost?', 'sales'),
    ('Do you have case studies in the healthcare sector?', 'sales'),
    ('We would like to discuss a multi-year agreement.', 'sales'),
    ('Is the premium support tier worth it for our size?', 'sales'),
    ('Can you share a feature comparison against your competitor?', 'sales'),
    ('What is your pricing for the API-only plan?', 'sales'),
    ('We need a formal quote for procurement approval.', 'sales'),
    ('Do you offer a startup programme?', 'sales'),
    ('Can we schedule a technical deep dive with your team?', 'sales'),
    ('What are the contract terms for annual prepayment?', 'sales'),
    ('Is there a minimum commitment for the enterprise plan?', 'sales'),
    ('Do you support custom contracts and redlines?', 'sales'),
    ('What would migrating 50,000 records cost us?', 'sales'),
    ('Can we get references from similar sized customers?', 'sales'),
    ('Is there a discount if we pay for two years upfront?', 'sales'),
    ('What is the cost of the compliance add-on?', 'sales'),
    ('We are expanding to a new region, what are the options?', 'sales'),
    ('Do you offer a money back guarantee?', 'sales'),
    ('Can you walk our team through the product next week?', 'sales'),
    ('What is your roadmap for the next two quarters?', 'sales'),
    ('Do you have a plan suited to agencies?', 'sales'),
    ('How much does the dedicated instance option cost?', 'sales'),
    ('We need pricing in GBP for our UK entity.', 'sales'),
    ('Can we extend our trial by two weeks to finish evaluating?', 'sales'),
    ('What is the uplift for adding premium SLA?', 'sales'),
    ('Please add my colleague as an admin on our workspace.', 'account management'),
    ('Can you merge our two organisation accounts?', 'account management'),
    ('I want to transfer ownership of the workspace to my manager.', 'account management'),
    ("Please remove a former employee's access immediately.", 'account management'),
    ('How do I rename our organisation?', 'account management'),
    ('We need SSO configured for our domain.', 'account management'),
    ('Please close our account at the end of the term.', 'account management'),
    ('Can you change the primary contact on our account?', 'account management'),
    ('I need to downgrade three users to read only.', 'account management'),
    ('Please export all our data before we migrate.', 'account management'),
    ('How do I set up a second workspace under the same org?', 'account management'),
    ('We need to add a domain to our verified list.', 'account management'),
    ('Please deactivate the account for our subsidiary.', 'account management'),
    ('Can you restore a user I deleted by mistake?', 'account management'),
    ('I want to enforce two-factor for everyone in the org.', 'account management'),
    ('How do I change the default role for new members?', 'account management'),
    ("Please remove the guest accounts from last year's project.", 'account management'),
    ('We need audit log retention extended to two years.', 'account management'),
    ('Can you move a project between our two workspaces?', 'account management'),
    ('I need to bulk invite 40 new team members.', 'account management'),
    ('Please update our organisation address on file.', 'account management'),
    ('How do I revoke all active sessions for a user?', 'account management'),
    ('We need to separate our two departments into different workspaces.', 'account management'),
    ('Can you grant our security team read access to the audit log?', 'account management'),
    ('Please remove my personal account but keep the team one.', 'account management'),
    ('I need to change the workspace URL slug.', 'account management'),
    ('How do I delegate admin rights while I am on leave?', 'account management'),
    ('Please archive the workspace but keep the data.', 'account management'),
    ('Can you reassign all items owned by a departed employee?', 'account management'),
    ('We want to restrict signups to our company domain.', 'account management'),
    ('How do I set up user provisioning with SCIM?', 'account management'),
    ('Please remove the trial workspace we no longer need.', 'account management'),
    ('Can you list everyone with admin access right now?', 'account management'),
    ('I need to change the notification settings for the whole org.', 'account management'),
    ('How do I recover access to an account whose owner left?', 'account management'),
    ('Can you disable public sharing across the organisation?', 'account management'),
    ('Please add a second owner to the organisation.', 'account management'),
    ('How do I see which users have not logged in for 90 days?', 'account management'),
    ('Can you enable the data residency setting for the EU?', 'account management'),
    ('Please delete all data for a user under GDPR.', 'account management'),
]

PERMUTATIONS = 4


def stratified_split(rows):
    """Alternate within each label so both halves see every class.

    Splitting this ordered dataset down the middle gives train and test DISJOINT
    label sets, which silently invalidates every number. Guarded below.
    """
    by_label: dict[str, list] = {}
    for row in rows:
        by_label.setdefault(row[1], []).append(row)
    train, test = [], []
    for group in by_label.values():
        train.extend(group[0::2])
        test.extend(group[1::2])
    return train, test


def predict(client, rows, tag="") -> list[Prediction]:
    out = []
    for i, (text, label) in enumerate(rows, 1):
        answer = client.ask({"ticket": text}, QUESTION).answers["route"]
        out.append(Prediction(answer.probabilities, label))
        print(f"  {tag} {i}/{len(rows)}", flush=True)
    return out


def show(name: str, rep) -> None:
    print(f"{name:<34} {rep.summary()}")


def main() -> None:
    backend = TransformersBackend.from_pretrained(MODEL)
    train, test = stratified_split(DATASET)
    labels = {label for _, label in DATASET}
    for name, split in (("train", train), ("test", test)):
        missing = labels - {label for _, label in split}
        if missing:
            raise SystemExit(f"{name} split is missing labels {sorted(missing)}")

    print(f"model={MODEL}  train={len(train)}  test={len(test)}\n")

    raw = Decider(backend)
    debiased = Decider(backend, calibration=Calibration(permutations=PERMUTATIONS, contextual=True))

    raw_test = predict(raw, test, "raw/test")
    deb_train = predict(debiased, train, "deb/train")
    deb_test = predict(debiased, test, "deb/test")

    fit = diagnose_temperature(deb_train)
    temperature = fit.temperature
    print(f"\ntemperature fit on train: {fit.summary()}")

    show("Phase 0 (raw)", report(raw_test))
    show(f"+ permutation(k={PERMUTATIONS}) + contextual", report(deb_test))
    show(f"+ fitted temperature T={temperature:.2f}", report(deb_test, temperature=temperature))

    print("\nmargin confidence (what a router gates on):")
    show("  Phase 0 (raw)", report(raw_test, confidence="margin"))
    show("  fully calibrated", report(deb_test, confidence="margin", temperature=temperature))

    print("\nselective accuracy, fully calibrated:")
    print("  threshold  coverage  accuracy")
    for point in report(deb_test, temperature=temperature).selective:
        acc = "n/a" if point.accuracy is None else f"{point.accuracy:.3f}"
        print(f"  {point.threshold:>9.2f}  {point.coverage:>8.3f}  {acc:>8}")


if __name__ == "__main__":
    main()
