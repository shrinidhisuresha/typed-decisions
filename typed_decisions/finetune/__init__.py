"""Outcome-calibrated fine-tuning: calibration in the weights instead of at inference.

Today a calibrated answer costs `permutations` passes, a null-state pass and a temperature
fitted per question family. This package trains a LoRA adapter so that ONE raw pass is
already calibrated:

- the loss is a proper scoring rule (log score) on the label-restricted softmax at the
  `Answer:` position -- exactly the distribution scoring.py serves, so calibration is
  part of the optimum rather than a patch on top;
- options are shown in a random order every time, so positional bias is penalised in the
  weights (replaces `permutations`);
- null-state rows are trained toward the family prior (replaces `contextual`).

data.py builds training rows, loss.py holds the objective, train.py fits the adapter and
eval.py compares it with the inference-time pipeline on datasets it never saw.
"""
