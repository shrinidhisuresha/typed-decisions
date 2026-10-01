# Latency per decision (BANKING77, 77 options, one query at a time)

| path | p50 ms | p95 ms | p99 ms | p50 µs |
|---|---:|---:|---:|---:|
| **probe end to end, MiniLM-L6 23M on CPU** | 2.27 | 4.19 | 7.17 | 2,274 |
| probe: embed query (0.6B) | 13.81 | 21.52 | 25.77 | 13,811 |
| probe: linear layer only (component) | 0.00 | 0.00 | 0.01 | 3 |
| probe end to end, Qwen3-Embedding-0.6B on the Apple GPU | 13.39 | 20.10 | 27.58 | 13,387 |
| **ModernBERT-base head end to end** | 10.56 | 16.80 | 19.14 | 10,564 |
| zero-shot 9B 4-bit (MLX), 4 ordering(s), retrieval, contextual | 1446.56 | 1588.00 | 1617.02 | 1,446,558 |
| zero-shot 9B 4-bit (MLX), 1 ordering(s), retrieval, contextual | 537.67 | 649.27 | 679.46 | 537,667 |

Apple M5 Pro, mps; 200 timed runs (30 for the 9B rows) after warm-up; perf_counter_ns with a device sync. The linear layer alone is a component figure, not a decision latency.
