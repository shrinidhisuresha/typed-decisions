# Outcome-calibrated fine-tuning: mlx-community/Qwen3.5-4B-4bit

Adapter: `adapters/mix2cp-4b-r16-s1`. Every task below is **held out**: the adapter never trained on its dataset. Each task is split in half per class; T is fitted on the train half and every number is on the test half.

## bitext_category (choice, 11 options, n=220 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| tuned | 1 | 0.091 | 0.016 | 0.910 | 19.999 | 0.001 | 0.909 |

## tickets_incident (noul, 2 options, n=150 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| tuned | 1 | 0.500 | 0.027 | 0.502 | 19.999 | 0.001 | 0.500 |

## tickets_priority (score, 3 options, n=75 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| tuned | 1 | 0.333 | 0.021 | 0.667 | 19.999 | 0.001 | 0.667 |

## bitext_telco_category (choice, 7 options, n=140 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| tuned | 1 | 0.143 | 0.019 | 0.858 | 19.999 | 0.001 | 0.857 |
