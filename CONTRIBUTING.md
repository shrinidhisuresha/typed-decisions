# Contributing

Thanks for helping. Issues and pull requests are welcome.

## Setup

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python -e '.[cpu,dev]'      # or [apple,dev] on a Mac
.venv/bin/python -m pytest tests/ -q -m "not slow"      # ~20 s, no model downloads beyond tiny test models
```

`-m slow` runs the end-to-end tests with real Qwen weights (a few GB).

## What a good change looks like

- **A test that fails without it.** Backends must also pass the "forked scores equal a
  whole-prompt forward" checks.
- **Numbers, not adjectives.** If you claim something is better or faster, add the script and
  the table (see `examples/` and `docs/`), including results that didn't work out.
- **Public data only.** Never commit private, customer or employer data, or API keys.
- **Stable wire format.** Don't break `/v1/ask`'s request and response shape without a
  major version bump.

## Scope

In scope: typed decisions (Choice, Score, Noul), calibration, serving backends, distillation,
retrieval, and the HTTP server. Out of scope: generating free text.
