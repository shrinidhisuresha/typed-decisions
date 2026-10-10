# Outcome-calibrated fine-tuning: mlx-community/Qwen3.5-4B-4bit

Adapter: `adapters/mix2cp-4b-r16`. Every task below is **held out**: the adapter never trained on its dataset. Each task is split in half per class; T is fitted on the train half and every number is on the test half.

## bitext_category (choice, 11 options, n=220 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 1 | 0.709 | 0.120 | 0.429 | 1.018 | 0.114 | 0.428 |
| posthoc | 4 + null | 0.755 | 0.123 | 0.373 | 0.687 | 0.090 | 0.351 |
| tuned | 1 | 0.727 | 0.154 | 0.410 | 1.628 | 0.083 | 0.386 |

**tuned (1 pass, T=1) vs posthoc (fitted T)**, paired 95% bootstrap: Brier diff [+0.000, +0.120], accuracy diff [-0.082, +0.023] -> gate: FAIL (Brier may be worse than teacher: diff CI upper +0.120 > +0.020; accuracy may drop: diff CI lower -0.082 < -0.020)

## tickets_incident (noul, 2 options, n=150 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 1 | 0.780 | 0.110 | 0.326 | 0.780 | 0.074 | 0.320 |
| posthoc | 4 + null | 0.780 | 0.091 | 0.314 | 0.702 | 0.067 | 0.304 |
| tuned | 1 | 0.800 | 0.075 | 0.302 | 1.162 | 0.065 | 0.296 |

**tuned (1 pass, T=1) vs posthoc (fitted T)**, paired 95% bootstrap: Brier diff [-0.057, +0.058], accuracy diff [-0.020, +0.060] -> gate: FAIL (Brier may be worse than teacher: diff CI upper +0.058 > +0.020)

## tickets_priority (score, 3 options, n=75 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 1 | 0.333 | 0.467 | 0.948 | 15.484 | 0.033 | 0.663 |
| posthoc | 4 + null | 0.467 | 0.035 | 0.614 | 1.912 | 0.095 | 0.626 |
| tuned | 1 | 0.453 | 0.186 | 0.687 | 4.331 | 0.047 | 0.633 |

**tuned (1 pass, T=1) vs posthoc (fitted T)**, paired 95% bootstrap: Brier diff [-0.032, +0.155], accuracy diff [-0.147, +0.120] -> gate: FAIL (Brier may be worse than teacher: diff CI upper +0.155 > +0.020; accuracy may drop: diff CI lower -0.147 < -0.020)

## bitext_telco_category (choice, 7 options, n=140 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 1 | 0.707 | 0.142 | 0.390 | 0.766 | 0.097 | 0.375 |
| posthoc | 4 + null | 0.743 | 0.221 | 0.392 | 0.549 | 0.096 | 0.337 |
| tuned | 1 | 0.829 | 0.045 | 0.252 | 1.098 | 0.030 | 0.249 |

**tuned (1 pass, T=1) vs posthoc (fitted T)**, paired 95% bootstrap: Brier diff [-0.150, -0.020], accuracy diff [+0.029, +0.150] -> gate: PASS (ok)
## Verdict (confidence penalty, 4B, 2026-10-06)

The same 4B recipe as `finetune_qwen3.5-4b_mix2.md`, plus `--confidence-penalty 0.3` on Noul and Score rows. 859 steps,
peak 13.2 GB.

| task | 4B mix2, T=1 acc / Brier (fitted T) | 4B mix2 + penalty | post-hoc, fitted T | gate |
|---|---|---|---|---|
| bitext_category | **0.800 / 0.321** (1.46) | 0.727 / 0.410 (1.63) | 0.755 / 0.351 | FAIL |
| bitext_telco_category | 0.800 / 0.263 (1.17) | **0.829 / 0.252** (1.10) | 0.743 / 0.337 | PASS |
| tickets_incident (Noul) | 0.800 / 0.337 (2.30) | 0.800 / **0.302 (1.16)** | 0.780 / 0.304 | FAIL: tie, CI [-0.057, +0.058] |
| tickets_priority (Score) | 0.453 / 0.726 (5.73) | 0.453 / **0.687** (4.33) | 0.467 / 0.626 | FAIL |

**1 of 4 pass at T=1, the same count as before. But the problem it targeted is mostly fixed:**

- **Noul is now calibrated at T=1** (fitted T 2.30 → 1.16). Its Brier ties the 5-pass post-hoc pipeline (0.302 vs 0.304);
  it fails the gate only because n=150 cannot rule out a 0.02 difference.
- **Score improved but is still overconfident** (T 5.73 → 4.33). A penalty of 0.3 is not enough for ordinal urgency at 4B.
- **bitext_category regressed by 7 points** (0.800 → 0.727), although Choice rows carry no penalty. Every run so far is a
  single seed, so this may be run-to-run variance rather than an effect of the penalty. Two more seeds would settle it
  (about 7 h at 4B).

### Seeds (2026-10-07): the recipe is unstable

Seeds 1 and 2 of exactly this run **collapsed** right after the learning-rate warmup peaked (1e-4 at step ~43):

| seed | loss at step 40 → 60 | end | result |
|---|---|---|---|
| 0 | 0.80 → 0.85 | 0.43–0.50 | the table above |
| 1 | 0.85 → 1.22 | 1.06 | near-uniform on every task: acc at chance, fitted T=20 (`finetune_qwen3.5-4b_mix2cp_s1.md`) |
| 2 | 0.94 → 1.46 | stopped at step 140 | the same collapse |

2 of 3 seeds fail, so the seed-0 numbers are not a reliable estimate of this recipe. Its 7-point bitext_category drop may be
a partial version of the same failure. The likely mechanism: one large update at peak LR, after which the
entropy penalty makes the uniform answer a stable basin, because it maximises H on Noul/Score rows. Gradient clipping at 1.0 did
not prevent it. A rerun needs a lower LR, the penalty switched on only after warmup, and an abort when loss jumps.

### Stability fix, 200-step test (2026-10-10)

Seeds 1 and 2 (the two that collapsed) rerun for 200 steps with `--lr 5e-5 --penalty-start -1` (penalty from the end of
warmup, step 42) `--abort-loss 1.1`. `--max-steps` now truncates the full-length schedule, so the warmup and peak happen
exactly as in a full run.

| seed | 20 | 40 | 60 | 80 | 100 | 140 | 200 |
|---|---|---|---|---|---|---|---|
| 1, original (lr 1e-4, penalty from 0) | 0.86 | 0.85 | **1.22** | 1.15 | 1.09 | 1.08 | collapsed |
| 1, fixed | 0.95 | 0.95 | 0.67 | 0.77 | 0.77 | 0.60 | 0.72 |
| 2, original | 0.73 | 0.94 | **1.46** | 1.50 | 1.34 | 1.39 | stopped |
| 2, fixed | 0.93 | 0.94 | 0.94 | 0.78 | 0.61 | 0.57 | 0.58 |

Both seeds get through the LR peak and fall to the range seed 0 reached; neither tripped the abort. Losses at
steps 20–60 are the starting level before learning, not a jump.
