# Design

How typed-decisions turns an open-weights model into calibrated, typed answers, and why each
piece is there. Measured results live in the README and `docs/`; this file explains the
mechanism. Section numbers are referenced from code comments.

## 1. The idea

Many software decisions are not open-ended text. They are closed questions: which of these
queues handles this ticket, how severe is it, is it urgent. For those, generating text and then
parsing it is the wrong tool:
- it is slow, because it runs the decode loop;
- it can fail to parse;
- it throws away the model's uncertainty.

Instead: give the model a **state** and a **typed question**, and read the probability of each
allowed answer directly from its logits at a single position.
- **No tokens are generated.**
- **The answer cannot be outside the allowed set.**
- **You get a probability distribution,** not just a label.

Three question types cover most needs:
- **Choice:** one of N options.
- **Score:** a position on an ordered scale.
- **Noul:** a calibrated yes/no.

## 2. Scoring math

**Choice over options o₁..oₙ.** Render the options with single-token labels (" A", " B", …),
checked at allocation time, never assumed. Run one forward pass, read the logits of those label
tokens at the answer position, and softmax over them. Nothing else in the vocabulary matters.
Beyond 26 options, or whenever option text matters, score each option's text as a continuation
and use its mean token log-probability (§3C covers retrieval for large sets).

**Noul.** The same pass with two labels: p(yes) / (p(yes) + p(no)). The number *is* the
probability; each caller picks its own threshold.

**Score.** Get the distribution over an ordered legend and return its expectation, Σ pᵢ·vᵢ. The
model never types a number, so the score moves smoothly between levels.

**Confidence.** A distribution collapsed to one number:
- margin, top-1 − top-2 (the default, best for routing);
- max probability;
- normalised max, (N·max − 1)/(N − 1);
- normalised entropy.

Pick one and hold it fixed; calibration is measured per definition.

## 3. Three ways to answer

### 3A. Decoder as a zero-shot scorer

Any open-weights instruct model, with no training, and questions defined at request time. It
inherits the model's positional and label biases, which is why §4 exists.

### 3B. Encoder heads trained on labels

When a question recurs and you have labelled outcomes, train a small head for that question:
- **A linear probe on frozen sentence embeddings:** trains in seconds, with millisecond inference.
- **A fine-tuned encoder classification head** (ModernBERT).

One head per question family (a fixed type, wording and option order). Training uses soft
cross-entropy (the log score): on the teacher's distribution, or on the one-hot outcome when
it's known. A head is served only after a paired-bootstrap gate shows it is no worse than the
zero-shot path on held-out outcomes (see the gate and shadow mode in the README).

### 3C. Retrieve, then score

For many plausible options:
1. Embed the state and every option.
2. Keep the top k.
3. Score only those with §3A.

Options cut by retrieval get probability 0. The answer is conditional on the shortlist, and
recall@k caps accuracy, so measure it.

## 4. Calibration

Raw probabilities from instruct models are biased and overconfident. The pipeline runs in this
order:
1. **Permutation:** rotate the option order and average the distributions mapped back to the
   original options. This cancels positional bias.
2. **Contextual calibration:** divide out the distribution the model gives with an empty state
   (cached per question). This cancels its standing preference for particular labels.
3. **Temperature:** p^(1/T), renormalised, with T fitted by negative log-likelihood on labelled
   data. This corrects residual over- or under-confidence.

**Zero probabilities** (rounded inputs, retrieval-cut options) are floored identically in the
fit and in the application. Otherwise the fit optimises a distribution that is never produced.
A fit pinned at its bound, or on a perfectly separable sample, is reported as untrustworthy and
not applied.

**Measure it, don't assume it:** ECE, Brier, reliability bins and the selective-accuracy curve
(coverage vs accuracy as the confidence threshold rises). The last is what a
confidence-gated router actually rides.

## 5. Serving and speed

- **Shared prefix.** Every question over one state shares that state's prefix, so it is prefilled
  once and each question forks from it. The state must be a strict prefix *at the token level*,
  not just in the string. The rendering makes question-first prompts impossible, and a test
  asserts that the forked scores equal scoring each prompt whole.
- **Backends.** The reference backend (Transformers) forks the KV cache explicitly. MLX does the
  same on Apple silicon with quantised weights. vLLM and SGLang are HTTP backends, and
  `examples/verify_backend.py` checks them against the reference on a live server.
- **The fastest path is §3B.** A probe on a small embedding model makes a decision in a few
  milliseconds on a CPU, orders of magnitude faster than any decoder pass. The decoder is the
  cold-start path for questions without labels.

## 6. What this deliberately does not do

**Constrained decoding** (grammar- or JSON-schema-guided generation). It guarantees valid output
but still generates, so it keeps the decode loop's latency, and it gives no probability
distribution over the answers. That is a different tool for a different problem.

## 7. Build path

The order the project was built in. Each phase is measured in the README.
1. **Phase 0, wrap:** typed questions → logit scoring → typed answers.
2. **Phase 1, calibrate:** permutation, contextual, temperature, and the measurement harness.
3. **Phase 2, serve:** shared-prefix forking, HTTP backends, cache hit-rate accounting.
4. **Phase 3, learn from outcomes:** traffic log, heads, promotion gate, shadow mode.
5. **Phase 4, large option sets:** retrieval before scoring.

## 8. Caveats

- **A zero-shot open model is not a purpose-trained decision model.** Expect to need labels, even
  a few per class, for the best accuracy.
- **Benchmarks here are small and mostly synthetic.** Measure on your own data before relying on
  any threshold.
- **Prefill cost grows with the state.** Keep the shared state compact.

## 9. Models

Every model used, its licence and its role are listed in
[`docs/MODELS_AND_DATA.md`](docs/MODELS_AND_DATA.md). The defaults:
- the Qwen3.5 family as the zero-shot scorer (0.8B, 4B, and 9B 4-bit on MLX);
- Qwen3-Embedding-0.6B for retrieval;
- all-MiniLM-L6-v2 for the fast probe;
- ModernBERT-base for trained heads.
