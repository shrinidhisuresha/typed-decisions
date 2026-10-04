# Typed decisions from logits, and 77-way intent in 2 ms

*4 October 2026*

## The problem

Software keeps asking LLMs closed questions: which team, how severe, is it spam. Generating text
and parsing it is slow, it can fail, and it throws away the model's uncertainty.

## The approach

- **Read the answer from the logits.** Give each option a single-token label and read its
  probability at one position. No decode loop, and the answer can't be outside the allowed set.
- **Calibrate it.** Average over rotated option orders (positional bias), divide out the empty-state
  prior (label bias), then fit a temperature. Report ECE, Brier and selective accuracy, not just
  accuracy.
- **Serve it cheaply.** Prefill the state once and fork the KV cache per question. On Apple
  silicon, MLX runs a 9B 4-bit model in about 5 GB.

## What we measured (BANKING77, 77 intents, official 3,080-query test set)

| labelled examples per class | accuracy | ECE | Brier | CPU latency, p50 |
|---:|---:|---:|---:|---:|
| 10 | 0.869 | 0.055 | 0.202 | 2.1 ms |
| 50 | 0.914 | 0.035 | 0.132 | 2.1 ms |
| all (10,003) | 0.933 | 0.046 | 0.111 | 2.1 ms |

The model is a linear probe on all-MiniLM-L6-v2 (23M). Across six embedding models (23M–596M),
accuracy at 50 per class stays between 0.913 and 0.929, while CPU latency goes from 2 ms to
52 ms:

| embedding model | params | accuracy (50/class) | CPU p50 |
|---|---:|---:|---:|
| all-MiniLM-L6-v2 | 23M | 0.914 | 2.1 ms |
| granite-embedding-30m-english | 30M | 0.921 | 2.0 ms |
| bge-small-en-v1.5 | 33M | 0.921 | 4.3 ms |
| e5-small-v2 | 33M | 0.913 | 4.0 ms |
| bge-base-en-v1.5 | 109M | 0.926 | 11.1 ms |
| Qwen3-Embedding-0.6B | 596M | 0.929 | 52.4 ms |

All latencies are one query end to end (embed, linear layer, softmax) on an Apple M5 Pro.

**The lesson:** zero-shot is the cold-start tool. Log outcomes, and once a question has a handful
of labels per class, a tiny probe is fast and accurate. typed-decisions automates that hand-over
with a paired-bootstrap promotion gate and shadow mode.

## What went wrong (and how we caught it)

- **A calibration bug.** An independent recomputation of our numbers found that the temperature
  step floored zero probabilities when fitting but not when applying. It was fixed before
  release: the same floor is now used in both places ([DESIGN.md §4](../../DESIGN.md#4-calibration)), and every number here was
  re-measured after the fix.
- **A vLLM finding.** On a vLLM CPU build, prefix-cache hits changed the returned logprobs. We
  default to an exact path and verify every backend against a reference.

## Try it

```bash
pip install 'typed-decisions[cpu]'
typed-decisions serve --preset cpu
```

- Code (Apache-2.0): https://github.com/shrinidhisuresha/typed-decisions
- Model: https://huggingface.co/Shrinidhisuresha/banking77-intent-probe-minilm
- Demo: https://huggingface.co/spaces/Shrinidhisuresha/typed-decisions-demo (it runs on free
  shared hardware, so expect it to be slower than the laptop numbers above)

## Limits

Everything was measured on one machine, and the headline is one public dataset (English,
banking). Zero-shot accuracy on fine-grained label sets is the weak spot, which is why the
labelled path exists. Measure on your own data.

*BANKING77 is CC BY 4.0: Casanueva et al. (2020), "Efficient Intent Detection with Dual Sentence
Encoders".*
