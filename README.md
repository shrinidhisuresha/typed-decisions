# typed-decisions

**Fast, calibrated typed decisions from open-weights models.** Ask closed questions about a
state (Choice, Score, or yes/no "Noul") and get a calibrated probability for every allowed
answer. Answers are read straight from model logits, or from a small head trained on your
labels. There is no text generation and no parsing, and an answer outside the allowed set is
impossible.

**Try it:** [live demo on Hugging Face Spaces](https://huggingface.co/spaces/Shrinidhisuresha/typed-decisions-demo)
· [BANKING77 probe model](https://huggingface.co/Shrinidhisuresha/banking77-intent-probe-minilm)
· [PyPI](https://pypi.org/project/typed-decisions/)
· [launch post](docs/blog/2026-10-04-typed-decisions.md)

## Results

All numbers come from scripts in `examples/`, on an Apple M5 Pro.

**77-way intent in about 2 ms on a laptop CPU.** A linear probe on the 23M-parameter
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) embedding model,
scored on [BANKING77](https://huggingface.co/datasets/PolyAI/banking77) (real banking queries, 77
intents, official 3,080-query test split):

| labelled examples per class | accuracy | ECE | Brier | latency, CPU p50 / p99 |
|---:|---:|---:|---:|---:|
| 10 | 0.869 | 0.055 | 0.202 | 2.1 / 3.1 ms |
| 50 | 0.914 | 0.035 | 0.132 | 2.1 / 3.1 ms |
| all (10,003) | **0.933** | 0.046 | 0.111 | 2.1 ms |

**Latency per decision** ([docs/latency_m5pro.md](docs/latency_m5pro.md)):

| path | p50 | p95 | p99 |
|---|---:|---:|---:|
| MiniLM probe, CPU | **2.3 ms** | 4.2 ms | 7.2 ms |
| ModernBERT-base head | 10.6 ms | 16.8 ms | 19.1 ms |
| Qwen3-Embedding-0.6B probe, Apple GPU | 13.4 ms | 20.1 ms | 27.6 ms |
| zero-shot Qwen3.5-9B 4-bit (MLX), 1 option order | 538 ms | 649 ms | 679 ms |
| zero-shot Qwen3.5-9B 4-bit (MLX), 4 option orders | 1,447 ms | 1,588 ms | 1,617 ms |

The probe's final linear layer takes about 3 µs. That's one component, not a decision latency.

**Embedding size vs accuracy and speed** ([docs/speed_sweep.md](docs/speed_sweep.md)): at 50
examples per class, models from 23M to 596M parameters land between 0.913 and 0.929, while CPU
latency rises from about 2 ms to 52 ms. Small embedders are the right default for the labelled
path.

**Shared-prefix serving** ([docs/bench_prefix_qwen3.5-0.8b_mps.md](docs/bench_prefix_qwen3.5-0.8b_mps.md)):
20 questions over one 2.6k-token state run 12.3× faster per question than re-reading the state
for each one.

## Quickstart

| flavour | install | serve |
|---|---|---|
| Apple silicon (MLX) | `pip install 'typed-decisions[apple]'` | `typed-decisions serve --preset apple` |
| NVIDIA GPU | the CUDA build of torch for your driver ([pytorch.org/get-started](https://pytorch.org/get-started/locally/)), then `pip install 'typed-decisions[cuda]'` | `typed-decisions serve --preset cuda` |
| CPU (no GPU) | `pip install 'typed-decisions[cpu]'` or `docker build -t typed-decisions:cpu .` | `typed-decisions serve --preset cpu` |
| client only (remote vLLM/SGLang) | `pip install typed-decisions` | `typed-decisions serve --backend vllm --url ...` |

Presets choose the zero-shot model:
- `apple`: Qwen3.5-9B 4-bit on MLX.
- `cuda`: Qwen3.5-4B.
- `cpu`: Qwen3.5-0.8B.

Each preset uses 4 option orders, contextual calibration, and retrieval for Choices over 26
options. Any flag you pass overrides the preset.

```bash
curl -s localhost:8080/v1/ask -d '{
  "state": {"ticket": "I was charged twice this month."},
  "questions": {
    "route":    {"type": "choice", "instructions": "Which team?",
                 "criteria": {"billing": "payments, refunds", "technical": "bugs, outages"}},
    "urgent":   {"type": "noul",   "instructions": "Is this urgent?"},
    "severity": {"type": "score",  "instructions": "How severe?",
                 "criteria": {"low": 1, "medium": 2, "high": 3}}}}'
```

## Python API

```python
from typed_decisions import Calibration, ChoiceQuestion, Decider, NoulQuestion, ScoreQuestion
from typed_decisions.backend_impls.transformers import TransformersBackend

decider = Decider(TransformersBackend.from_pretrained("Qwen/Qwen3.5-0.8B"),
                  calibration=Calibration(permutations=4, contextual=True))
result = decider.ask(
    {"ticket": "I was charged twice this month.", "tier": "enterprise"},
    {
        "route": ChoiceQuestion("Which team handles this?", ["billing", "technical", "sales"],
                                descriptions={"billing": "payments, invoices, refunds"}),
        "urgent": NoulQuestion("Is this urgent?"),
        "severity": ScoreQuestion("Rate severity.", {"low": 1.0, "medium": 3.0, "high": 5.0}),
    },
)
result.answers["route"].choice         # the most probable option. A 0.8B model zero-shot can
                                       # be wrong here (it said "technical"); see below
result.answers["route"].probabilities  # a probability for every option
result.answers["urgent"].noul          # P(yes), deliberately not thresholded
result.answers["severity"].score       # the expectation over the legend
result.usage.output_tokens             # 0
```

The question types:
- **Choice:** up to 255 options, with optional per-option descriptions.
- **Score:** a labelled legend, returning the probability-weighted value.
- **Noul:** yes/no, with optional meanings for yes and no.

Confidence is margin by default; max-probability, normalised max and entropy are also available.

## Zero-shot, calibrated

Without labels, an instruct model scores the question directly. The raw probabilities are biased,
so calibrate them (DESIGN.md §4):
- **Permutation** (`permutations=k`) rotates the option order to cancel positional bias.
- **Contextual calibration** (`contextual=True`) divides out the model's empty-state prior to
  cancel its preference for particular labels.
- **Temperature** is fitted on labelled data with `diagnose_temperature`, which refuses fits that
  are separable or at a bound.

`report()` gives ECE, Brier, reliability bins and the selective-accuracy curve. Measure on your
own data before gating on a confidence threshold. An uncalibrated confidence is worse than none,
because callers will trust it.

Zero-shot accuracy on fine-grained label sets (dozens of similar intents) is where open models
are weakest. That is what the labelled path below is for.

## From zero-shot to labels: the fast path

1. **Log traffic:** `typed-decisions serve --log traffic.jsonl`. Every response comes back with a
   `request_id`.
2. **Report what actually happened:** `POST /v1/outcomes {"request_id": ..., "outcomes": {"route": "billing"}}`.
3. **Train:** `typed-decisions distill train --log traffic.jsonl --family <key> --out heads/`
   trains an encoder head and gates it. A paired bootstrap must show it is no worse than the
   zero-shot path on held-out outcomes.
4. **Serve:** `typed-decisions serve --heads heads/` answers promoted families from their head in
   milliseconds; everything else falls back to the zero-shot model. `--shadow-heads` runs new
   heads beside it on live traffic first, and `typed-decisions distill shadow` gates them on real
   outcomes.

The embedding linear probe in the results above is in `examples/hf_banking77_probe.py`. Wiring it
in as a head type next to the ModernBERT head is on the roadmap.

## Backends

| backend | use | verification |
|---|---|---|
| `TransformersBackend` | reference; CPU, CUDA, Apple MPS | forked-cache scores equal whole-prompt scores (tests) |
| `MLXBackend` | Apple silicon, 4/8-bit quantised models | same fork-equals-whole tests on a tiny MLX model |
| `VLLMBackend` | serving | verified live against vLLM 0.11.0 (CPU) with `examples/verify_backend.py` |
| `SGLangBackend` | serving | mock-tested only. Run `verify_backend.py sglang ...` before relying on it |

A finding from that live check: on vLLM 0.11.0 (CPU), a prefix-cache hit changed the returned
logprobs. The vLLM backend therefore defaults to an exact path that bypasses the cache.
`cache_letters=True` re-enables it, and `verify_backend.py` tells you whether that's safe on your
deployment.

**Large option sets:** past 26 options a Choice is scored by option text. With
`--retriever` / `Decider(retriever=...)`, an embedding model first shortlists the top k options
(DESIGN.md §3C).

## HTTP server

```
POST /v1/ask        {"state": ..., "questions": {name: {type, instructions, criteria}}}
POST /v1/outcomes   {"request_id": ..., "outcomes": {name: label}}     (with --log)
GET  /v1/heads      distilled heads being served
GET  /v1/stats      prefix-cache hit rate
GET  /healthz
```

- **Binding:** the server binds to 127.0.0.1 by default.
- **When exposed:** use `--api-key` (or `TD_API_KEYS`), `--rate-limit-rpm` (429 with `Retry-After`)
  and `--max-queue` (529).
- **Docker:** the CPU image runs as a non-root user; see `Dockerfile` for memory notes.

## Limitations

- **Measured on one machine** (Apple M5 Pro). The headline accuracy is on one public dataset
  (BANKING77, English, banking domain). Measure on your own data.
- **SGLang is mock-tested only.** vLLM was verified on a CPU build, not on a GPU.
- **The labelled path needs labels.** It is the fastest and most accurate path, but only for
  questions you can collect outcomes for. Zero-shot is the cold-start path.
- **Calibration claims are per configuration.** Re-run `report()` when you change model,
  prompts or data.

## Models and data

Every model and dataset used, with its verified licence and role, is listed in
[docs/MODELS_AND_DATA.md](docs/MODELS_AND_DATA.md). Nothing is redistributed: weights and data
download from their original sources. BANKING77 is CC BY 4.0. If you use the results, cite
Casanueva et al. (2020), *Efficient Intent Detection with Dual Sentence Encoders*.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The fast test suite needs no GPU:

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python -e '.[cpu,dev]'
.venv/bin/python -m pytest tests/ -q -m "not slow"
```

Design rationale: [DESIGN.md](DESIGN.md). Licence: [Apache-2.0](LICENSE).
