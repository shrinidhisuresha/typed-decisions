# Outcome-calibrated fine-tuning: Qwen/Qwen3.5-0.8B

Adapter: `adapters/mix2-0.8b-r16`. Every task below is **held out**: the adapter never trained on its dataset. Each task is split in half per class; T is fitted on the train half and every number is on the test half.

## bitext_category (choice, 11 options, n=220 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 1 | 0.091 | 0.482 | 1.170 | 19.999 | 0.015 | 0.908 |
| posthoc | 4 + null | 0.168 | 0.010 | 0.903 | 2.132 | 0.047 | 0.901 |
| tuned | 1 | 0.718 | 0.104 | 0.406 | 1.267 | 0.074 | 0.390 |

**tuned (1 pass, T=1) vs posthoc (fitted T)**, paired 95% bootstrap: Brier diff [-0.571, -0.419], accuracy diff [+0.477, +0.623] -> gate: PASS (ok)

## tickets_incident (noul, 2 options, n=150 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 1 | 0.500 | 0.338 | 0.683 | 8.770 | 0.049 | 0.494 |
| posthoc | 4 + null | 0.627 | 0.100 | 0.479 | 0.155 | 0.054 | 0.437 |
| tuned | 1 | 0.780 | 0.172 | 0.359 | 2.674 | 0.077 | 0.312 |

**tuned (1 pass, T=1) vs posthoc (fitted T)**, paired 95% bootstrap: Brier diff [-0.161, +0.017], accuracy diff [+0.073, +0.227] -> gate: PASS (ok)

## tickets_priority (score, 3 options, n=75 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 1 | 0.347 | 0.345 | 0.876 | 19.999 | 0.004 | 0.667 |
| posthoc | 4 + null | 0.360 | 0.063 | 0.676 | 8.029 | 0.016 | 0.666 |
| tuned | 1 | 0.427 | 0.235 | 0.712 | 3.838 | 0.039 | 0.635 |

**tuned (1 pass, T=1) vs posthoc (fitted T)**, paired 95% bootstrap: Brier diff [-0.061, +0.149], accuracy diff [-0.080, +0.213] -> gate: FAIL (Brier may be worse than teacher: diff CI upper +0.149 > +0.020; accuracy may drop: diff CI lower -0.080 < -0.020)

## bitext_telco_category (choice, 7 options, n=140 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 1 | 0.214 | 0.283 | 1.045 | 19.999 | 0.052 | 0.856 |
| posthoc | 4 + null | 0.300 | 0.103 | 0.828 | 0.942 | 0.091 | 0.827 |
| tuned | 1 | 0.621 | 0.148 | 0.487 | 1.422 | 0.091 | 0.466 |

**tuned (1 pass, T=1) vs posthoc (fitted T)**, paired 95% bootstrap: Brier diff [-0.434, -0.246], accuracy diff [+0.221, +0.421] -> gate: PASS (ok)
## Verdict (Score fix, 2026-10-04)

Training: mixture `mix2`. This is `mix1` plus Amazon star ratings (Score, 5- and 3-level), Civil Comments toxicity
(Score, 4-level) and Civil Comments soft Noul (target = fraction of annotators who said toxic).
`--scale 0.8`, 7,195 rows, RPS weight 1.0 on Score rows, 900 steps.

Accuracy and Brier at T=1, one pass:

| task | post-hoc (fitted T) | mix1 adapter | mix2 adapter | gate (mix2 vs post-hoc) |
|---|---|---|---|---|
| bitext_category | 0.168 / 0.901 | 0.668 / 0.478 | **0.718 / 0.406** | PASS |
| bitext_telco_category | 0.300 / 0.827 | **0.679 / 0.483** | 0.621 / 0.488 | PASS |
| tickets_incident (Noul) | 0.627 / 0.437 | 0.800 / 0.371 | 0.780 / **0.359** | **PASS** (was a 0.001 miss) |
| tickets_priority (Score) | 0.360 / 0.666 | 0.387 / 0.824 | **0.427 / 0.712** | FAIL: CI spans 0 at n=75 |

**3 of 4 held-out tasks pass. Score improved but is not fixed.**

- **Overconfidence on Score dropped by more than half:** fitted T went from 9.1 to 3.8 and ECE from 0.30 to 0.24. With a fitted T, the
  adapter's Brier (0.635) is the best of every config on priority. At T=1 it is still worse than post-hoc, which fits
  T on this task's labels, but the paired CI [-0.061, +0.149] cannot separate them at n=75.
- **The mode collapse moved rather than went away:** 79% of tickets are now "medium" (previously 81% "high"). The
  0.8B model still barely reads urgency (accuracy 0.43 vs chance 0.33). A larger scorer is the next lever, not more data at this size.
- **Choice and Noul mostly held.** Telco dropped 6 points as the Choice share of the mixture fell.
