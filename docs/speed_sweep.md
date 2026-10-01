# Speed sweep: embedding size vs accuracy and latency (BANKING77, 77 intents)

Linear probe on k examples per class from the official train split; metrics on the official 3,080-row test split. Latency is one query end to end (embed + linear + softmax), Apple M5 Pro, 200 runs after warm-up.

| embedding model | params | k | acc | ECE | Brier | mps p50 / p99 ms | cpu p50 / p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | 23M | 10 | 0.869 | 0.055 | 0.202 | 4.4 / 6.3 | 2.1 / 3.1 |
| all-MiniLM-L6-v2 | 23M | 50 | 0.914 | 0.035 | 0.132 | 4.4 / 6.3 | 2.1 / 3.1 |
| granite-embedding-30m-english | 30M | 10 | 0.856 | 0.077 | 0.220 | 3.9 / 6.0 | 2.0 / 3.0 |
| granite-embedding-30m-english | 30M | 50 | 0.921 | 0.066 | 0.130 | 3.9 / 6.0 | 2.0 / 3.0 |
| bge-small-en-v1.5 | 33M | 10 | 0.881 | 0.077 | 0.188 | 5.3 / 9.2 | 4.3 / 7.6 |
| bge-small-en-v1.5 | 33M | 50 | 0.921 | 0.061 | 0.126 | 5.3 / 9.2 | 4.3 / 7.6 |
| e5-small-v2 | 33M | 10 | 0.844 | 0.144 | 0.255 | 5.7 / 8.3 | 4.0 / 9.4 |
| e5-small-v2 | 33M | 50 | 0.913 | 0.122 | 0.158 | 5.7 / 8.3 | 4.0 / 9.4 |
| bge-base-en-v1.5 | 109M | 10 | 0.891 | 0.076 | 0.172 | 8.4 / 11.3 | 11.1 / 19.9 |
| bge-base-en-v1.5 | 109M | 50 | 0.926 | 0.052 | 0.116 | 8.4 / 11.3 | 11.1 / 19.9 |
| Qwen3-Embedding-0.6B | 596M | 10 | 0.876 | 0.076 | 0.191 | 19.0 / 29.5 | 52.4 / 114.8 |
| Qwen3-Embedding-0.6B | 596M | 50 | 0.929 | 0.058 | 0.119 | 19.0 / 29.5 | 52.4 / 114.8 |
