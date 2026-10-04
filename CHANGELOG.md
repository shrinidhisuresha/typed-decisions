# Changelog

## 0.1.0 (2026-10-04)

First public release.

- **Typed decisions** (Choice, Score, Noul) by logit scoring, with zero generated tokens.
- **Calibration:** permutation, contextual and temperature calibration, plus ECE, Brier and
  selective-accuracy reports.
- **Fast labelled path:** a linear probe on a 23M embedding model: 0.933 on BANKING77, about 2 ms per decision on CPU.
- **Backends:** Transformers (reference), MLX (Apple silicon, quantised), vLLM (verified live)
  and SGLang (mock-tested).
- **Serving:** shared-prefix serving, and Choice over up to 255 options via option-text scoring
  and retrieval.
- **Distillation:** traffic logging, distilled heads, a paired-bootstrap promotion gate and
  shadow mode.
- **Packaging:** flavours (`cpu`, `cuda`, `apple`, `train`), the `typed-decisions` command,
  presets, and a CPU Docker image.
