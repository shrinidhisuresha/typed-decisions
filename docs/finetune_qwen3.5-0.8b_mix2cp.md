# Outcome-calibrated fine-tuning: Qwen/Qwen3.5-0.8B

Adapter: `adapters/mix2cp-0.8b-r16`. Every task below is **held out**: the adapter never trained on its dataset. Each task is split in half per class; T is fitted on the train half and every number is on the test half.

## bitext_category (choice, 11 options, n=220 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 1 | 0.091 | 0.482 | 1.170 | 19.999 | 0.015 | 0.908 |
| posthoc | 4 + null | 0.168 | 0.010 | 0.903 | 2.132 | 0.047 | 0.901 |
| tuned | 1 | 0.727 | 0.080 | 0.367 | 1.132 | 0.056 | 0.359 |

**tuned (1 pass, T=1) vs posthoc (fitted T)**, paired 95% bootstrap: Brier diff [-0.605, -0.462], accuracy diff [+0.486, +0.632] -> gate: PASS (ok)

## tickets_incident (noul, 2 options, n=150 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 1 | 0.500 | 0.338 | 0.683 | 8.770 | 0.049 | 0.494 |
| posthoc | 4 + null | 0.627 | 0.100 | 0.479 | 0.155 | 0.054 | 0.437 |
| tuned | 1 | 0.807 | 0.126 | 0.329 | 1.972 | 0.058 | 0.306 |

**tuned (1 pass, T=1) vs posthoc (fitted T)**, paired 95% bootstrap: Brier diff [-0.183, -0.022], accuracy diff [+0.107, +0.253] -> gate: PASS (ok)

## tickets_priority (score, 3 options, n=75 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 1 | 0.347 | 0.345 | 0.876 | 19.999 | 0.004 | 0.667 |
| posthoc | 4 + null | 0.360 | 0.063 | 0.676 | 8.029 | 0.016 | 0.666 |
| tuned | 1 | 0.440 | 0.148 | 0.675 | 2.706 | 0.042 | 0.634 |

**tuned (1 pass, T=1) vs posthoc (fitted T)**, paired 95% bootstrap: Brier diff [-0.079, +0.094], accuracy diff [-0.067, +0.240] -> gate: FAIL (Brier may be worse than teacher: diff CI upper +0.094 > +0.020; accuracy may drop: diff CI lower -0.067 < -0.020)

## bitext_telco_category (choice, 7 options, n=140 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 1 | 0.214 | 0.283 | 1.045 | 19.999 | 0.052 | 0.856 |
| posthoc | 4 + null | 0.300 | 0.103 | 0.828 | 0.942 | 0.091 | 0.827 |
| tuned | 1 | 0.650 | 0.134 | 0.489 | 1.446 | 0.052 | 0.467 |

**tuned (1 pass, T=1) vs posthoc (fitted T)**, paired 95% bootstrap: Brier diff [-0.436, -0.235], accuracy diff [+0.243, +0.450] -> gate: PASS (ok)
## Verdict (confidence penalty, 2026-10-06)

The same as `mix2` (scale 0.8, RPS 1.0), plus `--confidence-penalty 0.3`: the loss subtracts 0.3·H(p) on Noul and Score rows
only.

| task | mix2, T=1 acc / Brier (fitted T) | mix2 + penalty | gate |
|---|---|---|---|
| bitext_category | 0.718 / 0.406 (1.27) | **0.727 / 0.367** (1.13) | PASS |
| bitext_telco_category | 0.621 / 0.488 (1.42) | **0.650** / 0.489 (1.45) | PASS |
| tickets_incident (Noul) | 0.780 / 0.359 (2.67) | **0.807 / 0.329** (1.97) | PASS |
| tickets_priority (Score) | 0.427 / 0.712 (3.84) | **0.440 / 0.675** (2.71) | FAIL: CI spans 0, n=75 |

3 of 4 pass, as before. Every task is level or better, and the fitted T moves toward 1 on both penalised shapes.
