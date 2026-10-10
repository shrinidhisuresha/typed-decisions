# Outcome-calibrated fine-tuning: mlx-community/Qwen3.5-4B-4bit

Adapter: `adapters/mix2cps-4b-r16-s1`. Every task below is **held out**: the adapter never trained on its dataset. Each task is split in half per class; T is fitted on the train half and every number is on the test half.

## bitext_category (choice, 11 options, n=220 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| tuned | 1 | 0.732 | 0.162 | 0.422 | 1.436 | 0.100 | 0.379 |

## tickets_incident (noul, 2 options, n=150 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| tuned | 1 | 0.800 | 0.107 | 0.310 | 1.597 | 0.030 | 0.289 |

## tickets_priority (score, 3 options, n=75 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| tuned | 1 | 0.413 | 0.270 | 0.735 | 4.954 | 0.047 | 0.639 |

## bitext_telco_category (choice, 7 options, n=140 test)

| config | passes | acc | ECE (T=1) | Brier (T=1) | fitted T | ECE (fitted) | Brier (fitted) |
|---|---:|---:|---:|---:|---:|---:|---:|
| tuned | 1 | 0.821 | 0.064 | 0.255 | 1.091 | 0.049 | 0.255 |
