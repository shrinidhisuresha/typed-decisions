# Outcome-calibrated fine-tuning: mlx-community/Qwen3.5-4B-4bit

Adapter: `adapters/mix2-4b-r16`. Every task below is **held out**: the adapter never trained on its dataset. Each task is split in half per class; T is fitted on the train half and every number is on the test half.

## bitext_category (choice, 11 options, n=220 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 1 | 0.709 | 0.120 | 0.429 | 1.018 | 0.114 | 0.428 |
| posthoc | 4 + null | 0.755 | 0.123 | 0.373 | 0.687 | 0.090 | 0.351 |
| tuned | 1 | 0.800 | 0.120 | 0.321 | 1.456 | 0.076 | 0.310 |

**tuned (1 pass, T=1) vs posthoc (fitted T)**, paired 95% bootstrap: Brier diff [-0.078, +0.020], accuracy diff [-0.005, +0.091] -> gate: FAIL (Brier may be worse than teacher: diff CI upper +0.020 > +0.020)

## tickets_incident (noul, 2 options, n=150 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 1 | 0.780 | 0.110 | 0.326 | 0.780 | 0.074 | 0.320 |
| posthoc | 4 + null | 0.780 | 0.091 | 0.314 | 0.702 | 0.067 | 0.304 |
| tuned | 1 | 0.800 | 0.159 | 0.337 | 2.296 | 0.034 | 0.285 |

**tuned (1 pass, T=1) vs posthoc (fitted T)**, paired 95% bootstrap: Brier diff [-0.044, +0.120], accuracy diff [-0.020, +0.060] -> gate: FAIL (Brier may be worse than teacher: diff CI upper +0.120 > +0.020)

## tickets_priority (score, 3 options, n=75 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 1 | 0.333 | 0.467 | 0.948 | 15.484 | 0.033 | 0.663 |
| posthoc | 4 + null | 0.467 | 0.035 | 0.614 | 1.912 | 0.095 | 0.626 |
| tuned | 1 | 0.453 | 0.236 | 0.726 | 5.726 | 0.047 | 0.634 |

**tuned (1 pass, T=1) vs posthoc (fitted T)**, paired 95% bootstrap: Brier diff [-0.010, +0.209], accuracy diff [-0.147, +0.120] -> gate: FAIL (Brier may be worse than teacher: diff CI upper +0.209 > +0.020; accuracy may drop: diff CI lower -0.147 < -0.020)

## bitext_telco_category (choice, 7 options, n=140 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 1 | 0.707 | 0.142 | 0.390 | 0.766 | 0.097 | 0.375 |
| posthoc | 4 + null | 0.743 | 0.221 | 0.392 | 0.549 | 0.096 | 0.337 |
| tuned | 1 | 0.800 | 0.082 | 0.263 | 1.174 | 0.065 | 0.261 |

**tuned (1 pass, T=1) vs posthoc (fitted T)**, paired 95% bootstrap: Brier diff [-0.142, -0.007], accuracy diff [-0.007, +0.129] -> gate: PASS (ok)
## Verdict (4B, 2026-10-05)

Model: `mlx-community/Qwen3.5-4B-4bit`, QLoRA r=16 via `finetune/train_mlx.py`, mixture `mix2` (scale 0.8,
RPS 1.0). 6,871 of 7,195 rows fit the 256-token cap. 859 steps, about 3.5 h on an M5 Pro, peak 13.2 GB. Baselines are the
same 4-bit model.

Accuracy and Brier, one pass at T=1 for tuned, vs post-hoc at its fitted T (the gate's comparison):

| task | post-hoc, fitted T | tuned, T=1 | tuned, fitted T | gate |
|---|---|---|---|---|
| bitext_category | 0.755 / 0.351 | **0.800 / 0.321** | 0.310 (T=1.46) | FAIL by 0.0004 (Brier CI upper +0.0204) |
| bitext_telco_category | 0.743 / 0.337 | **0.800 / 0.263** | 0.261 (T=1.17) | PASS |
| tickets_incident (Noul) | 0.780 / 0.304 | 0.800 / 0.337 | **0.285** (T=2.30) | FAIL: overconfident |
| tickets_priority (Score) | 0.467 / 0.626 | 0.453 / 0.726 | 0.634 (T=5.73) | FAIL: overconfident |

**Step-3 criterion: NO-GO as stated (1 of 4 tasks pass at T=1). As a one-pass model with a fitted T, it is the best config on 3 of 4 tasks.**

- **Accuracy:** the adapter matches or beats the 5-pass post-hoc pipeline on every task except priority, which is level within
  noise (0.453 vs 0.467, n=75). Choice is strong everywhere: telco 0.743 → 0.800 with Brier 0.337 → 0.263.
- **The "no fitted T" claim does not survive at 4B.** Choice is close to calibrated (T 1.2–1.5), but Noul
  (T=2.3) and Score (T=5.7) are overconfident on unseen domains, as they were at 0.8B. A proper scoring rule
  calibrates on the training distribution; these held-out tasks are a different distribution.
- **What works today:** adapter + a per-family temperature fitted on a few hundred labelled rows. That is one pass instead of
  five, with the best Brier on bitext_category, telco and incident. Priority stays within noise of post-hoc.
- **Next levers, cheapest first:** add a confidence penalty or label smoothing only on Noul/Score rows, and add
  more held-out-style ordinal data (severity, urgency). Then re-test whether T=1 holds.
