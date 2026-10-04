# Outcome-calibrated fine-tuning: Qwen/Qwen3.5-0.8B

Adapter: `adapters/b77-0.8b-r16`. Every task below is **held out**: the adapter never trained on its dataset. Each task is split in half per class; T is fitted on the train half and every number is on the test half.

## bitext_category (choice, 11 options, n=220 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 1 | 0.091 | 0.482 | 1.170 | 19.999 | 0.015 | 0.908 |
| posthoc | 4 + null | 0.168 | 0.010 | 0.903 | 2.132 | 0.047 | 0.901 |
| tuned | 1 | 0.777 | 0.100 | 0.301 | 1.299 | 0.047 | 0.288 |

**tuned (1 pass, T=1) vs posthoc (fitted T)**, paired 95% bootstrap: Brier diff [-0.673, -0.519], accuracy diff [+0.536, +0.682] -> gate: PASS (ok)

## tickets_incident (noul, 2 options, n=150 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 1 | 0.500 | 0.338 | 0.683 | 8.770 | 0.049 | 0.494 |
| posthoc | 4 + null | 0.627 | 0.100 | 0.479 | 0.155 | 0.054 | 0.437 |
| tuned | 1 | 0.700 | 0.211 | 0.506 | 4.051 | 0.079 | 0.437 |

**tuned (1 pass, T=1) vs posthoc (fitted T)**, paired 95% bootstrap: Brier diff [-0.005, +0.153], accuracy diff [+0.013, +0.133] -> gate: FAIL (Brier may be worse than teacher: diff CI upper +0.153 > +0.020)

## tickets_priority (score, 3 options, n=75 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw | 1 | 0.347 | 0.345 | 0.876 | 19.999 | 0.004 | 0.667 |
| posthoc | 4 + null | 0.360 | 0.063 | 0.676 | 8.029 | 0.016 | 0.666 |
| tuned | 1 | 0.373 | 0.365 | 0.856 | 6.503 | 0.080 | 0.646 |

**tuned (1 pass, T=1) vs posthoc (fitted T)**, paired 95% bootstrap: Brier diff [+0.040, +0.342], accuracy diff [-0.040, +0.067] -> gate: FAIL (Brier may be worse than teacher: diff CI upper +0.342 > +0.020; accuracy may drop: diff CI lower -0.040 < -0.020)
## Verdict (first run, 2026-10-04)

Training: BANKING77 only, 30/class, 2,562 rows (Choice + Noul + 10% null-state), LoRA r=16, 1 epoch,
321 steps, 17 min on an M5 Pro (gradient checkpointing, 2.4 GB).

**Step-2 criterion: NO-GO. It passes on 1 of 3 held-out datasets; the criterion needs 2.**

- **Choice transfers strongly.** bitext_category goes from 0.168 to 0.777 accuracy and Brier from 0.90 to 0.30 in one pass at
  T=1. But bitext is customer intents, close to BANKING77's domain, and Choice is the format it trained on.
  This is near transfer, not a new schema.
- **Accuracy transfers to Noul; calibration does not.** On tickets_incident, accuracy rises from 0.627 to 0.700 but the model is
  overconfident at T=1 (ECE 0.21, fitted T=4.1). With a fitted T it ties post-hoc on Brier.
- **Score was never trained, and it shows.** On tickets_priority, accuracy is flat and the model is overconfident (T=6.5).
- The fix in the plan follows directly: a multi-dataset mixture that includes Score rows and Noul from other
  domains, so that the proper scoring rule is optimised over the shapes it is tested on.
