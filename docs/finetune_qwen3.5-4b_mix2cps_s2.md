# Outcome-calibrated fine-tuning: mlx-community/Qwen3.5-4B-4bit

Adapter: `adapters/mix2cps-4b-r16-s2`. Every task below is **held out**: the adapter never trained on its dataset. Each task is split in half per class; T is fitted on the train half and every number is on the test half.

## bitext_category (choice, 11 options, n=220 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| tuned | 1 | 0.764 | 0.144 | 0.379 | 1.432 | 0.057 | 0.349 |

## tickets_incident (noul, 2 options, n=150 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| tuned | 1 | 0.787 | 0.132 | 0.323 | 1.654 | 0.054 | 0.290 |

## tickets_priority (score, 3 options, n=75 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| tuned | 1 | 0.467 | 0.220 | 0.773 | 6.406 | 0.066 | 0.652 |

## bitext_telco_category (choice, 7 options, n=140 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| tuned | 1 | 0.764 | 0.099 | 0.285 | 1.433 | 0.120 | 0.280 |
