# Outcome-calibrated fine-tuning: Qwen/Qwen3.5-0.8B

Adapter: `adapters/mix1-0.8b-r16`. Every task below is **held out**: the adapter never trained on its dataset. Each task is split in half per class; T is fitted on the train half and every number is on the test half.

## bitext_category (choice, 11 options, n=220 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 1 | 0.091 | 0.482 | 1.170 | 19.999 | 0.015 | 0.908 |
| posthoc | 4 + null | 0.168 | 0.010 | 0.903 | 2.132 | 0.047 | 0.901 |
| tuned | 1 | 0.668 | 0.144 | 0.478 | 1.264 | 0.071 | 0.459 |

**tuned (1 pass, T=1) vs posthoc (fitted T)**, paired 95% bootstrap: Brier diff [-0.504, -0.339], accuracy diff [+0.427, +0.568] -> gate: PASS (ok)

## tickets_incident (noul, 2 options, n=150 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 1 | 0.500 | 0.338 | 0.683 | 8.770 | 0.049 | 0.494 |
| posthoc | 4 + null | 0.627 | 0.100 | 0.479 | 0.155 | 0.054 | 0.437 |
| tuned | 1 | 0.800 | 0.160 | 0.371 | 2.593 | 0.062 | 0.327 |

**tuned (1 pass, T=1) vs posthoc (fitted T)**, paired 95% bootstrap: Brier diff [-0.143, +0.021], accuracy diff [+0.113, +0.240] -> gate: FAIL (Brier may be worse than teacher: diff CI upper +0.021 > +0.020)

## tickets_priority (score, 3 options, n=75 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 1 | 0.347 | 0.345 | 0.876 | 19.999 | 0.004 | 0.667 |
| posthoc | 4 + null | 0.360 | 0.063 | 0.676 | 8.029 | 0.016 | 0.666 |
| tuned | 1 | 0.387 | 0.303 | 0.824 | 9.063 | 0.092 | 0.657 |

**tuned (1 pass, T=1) vs posthoc (fitted T)**, paired 95% bootstrap: Brier diff [+0.033, +0.293], accuracy diff [-0.040, +0.093] -> gate: FAIL (Brier may be worse than teacher: diff CI upper +0.293 > +0.020; accuracy may drop: diff CI lower -0.040 < -0.020)

## bitext_telco_category (choice, 7 options, n=140 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 1 | 0.214 | 0.283 | 1.045 | 19.999 | 0.052 | 0.856 |
| posthoc | 4 + null | 0.300 | 0.103 | 0.828 | 0.942 | 0.091 | 0.827 |
| tuned | 1 | 0.679 | 0.131 | 0.483 | 1.449 | 0.042 | 0.467 |

**tuned (1 pass, T=1) vs posthoc (fitted T)**, paired 95% bootstrap: Brier diff [-0.446, -0.242], accuracy diff [+0.279, +0.479] -> gate: PASS (ok)
## Verdict (mixture run, 2026-10-04)

Training: mixture `mix1`. That is 6,337 rows from BANKING77, AG News and TREC (Choice), Enron spam, SUBJ and SST-2
(Noul, some with negated wording) and SST-5 (Score on a 5- and a 3-level legend), plus 10% null-state rows.
LoRA r=16, 1 epoch, 793 steps, 44 min on an M5 Pro. None of the four eval datasets is in the mixture.

Accuracy and Brier at T=1, one pass:

| task | post-hoc (5 passes, fitted T) | BANKING77-only adapter | mixture adapter | gate (mixture vs post-hoc) |
|---|---|---|---|---|
| bitext_category | 0.168 / 0.901 | **0.777 / 0.301** | 0.668 / 0.478 | PASS |
| bitext_telco_category | 0.300 / 0.827 | 0.679 / 0.515 | **0.679 / 0.483** | PASS |
| tickets_incident (Noul) | 0.627 / 0.437 | 0.700 / 0.506 | **0.800 / 0.371** | FAIL by 0.001 (Brier CI upper +0.021 vs margin +0.020) |
| tickets_priority (Score) | 0.360 / 0.666 | 0.373 / 0.857 | 0.387 / 0.824 | FAIL: overconfident, fitted T=9.1 |

**Step-2 criterion (≥2 held-out tasks pass): GO, with Score still open.**

- **The mixture fixed Noul.** Accuracy went from 0.700 to 0.800 and the fitted T from 4.1 to 2.6. Its mean Brier beats post-hoc, but the
  CI misses the margin by 0.001 at n=150.
- **Choice generalises across domains either way.** Telco, a domain neither adapter trained on, goes from 0.300 to 0.679. The
  mixture costs 11 points on bitext_category, which was the BANKING77 adapter's near-domain task. That is the price of
  diluting the intent data from 100% to about a quarter of the rows.
- **Score does not transfer from sentiment to urgency.** The model is still badly overconfident (T=9). Ticket priority
  labels are also noisy: the dataset is synthetic and its labels are known to be unreliable. n=75 is small. The next step is ordinal data closer to
  "how severe or urgent", or an RPS loss on Score rows, before Score is claimed.
