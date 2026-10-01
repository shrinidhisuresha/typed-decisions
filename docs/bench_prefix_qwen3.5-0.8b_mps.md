# Prefix-sharing benchmark: Qwen/Qwen3.5-0.8B on mps

Median of 3 runs after one warm-up. `ms/q` is wall time per question.

| state tokens | k | shared ms/q | naive ms/q | speedup | prefix hit rate |
|---:|---:|---:|---:|---:|---:|
| 335 | 1 | 385.0 | 323.3 | 0.84x | 0.0% |
| 335 | 20 | 81.1 | 325.0 | 4.01x | 86.2% |
| 2639 | 1 | 2575.6 | 2509.8 | 0.97x | 0.0% |
| 2639 | 20 | 207.3 | 2540.5 | 12.26x | 93.8% |
