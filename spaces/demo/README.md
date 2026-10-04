---
title: typed-decisions demo
emoji: ⚡
colorFrom: indigo
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
models:
- Shrinidhisuresha/banking77-intent-probe-minilm
- sentence-transformers/all-MiniLM-L6-v2
datasets:
- PolyAI/banking77
---

77-way BANKING77 intent classification in a few milliseconds on CPU, with a calibrated
probability for every intent. It's a linear probe on all-MiniLM-L6-v2, built with
[typed-decisions](https://github.com/shrinidhisuresha/typed-decisions).

Data: BANKING77 (CC BY 4.0), Casanueva et al. (2020), *Efficient Intent Detection with Dual
Sentence Encoders*.
