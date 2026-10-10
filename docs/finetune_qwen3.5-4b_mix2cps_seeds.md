# 4B stable recipe, 3 seeds

`train_mlx.py --mixture mix2 --scale 0.8 --rps-weight 1.0 --confidence-penalty 0.3 --lr 5e-5 --penalty-start -1 --abort-loss 1.1`,
seeds 0–2, Qwen3.5-4B-4bit. All three trained without collapse or abort. Tuned numbers are one pass at T=1; the post-hoc
pipeline (4 orderings + null pass) is shown at its fitted T, the gate's comparison. Mean ± sd over seeds.

| task | tuned acc | tuned Brier | fitted T | gate passes | no-penalty run (1 seed) | post-hoc, fitted T |
|---|---|---|---|---|---|---|
| bitext_category | 0.750 ± 0.016 | 0.402 ± 0.022 | 1.46 ± 0.05 | 0/3 | 0.800 / 0.321 | 0.755 / 0.351 |
| bitext_telco_category | 0.779 ± 0.038 | 0.302 ± 0.058 | 1.34 ± 0.22 | 1/3 | 0.800 / 0.263 | 0.743 / 0.337 |
| tickets_incident | 0.800 ± 0.013 | 0.313 ± 0.009 | 1.62 ± 0.03 | 0/3 | 0.800 / 0.337 | 0.780 / 0.304 |
| tickets_priority | 0.453 ± 0.035 | 0.745 ± 0.024 | 5.71 ± 0.73 | 0/3 | 0.453 / 0.726 | 0.467 / 0.626 |

Per seed (acc / Brier / fitted T / gate):

- bitext_category: s0 0.755 / 0.403 / 1.53 / fail; s1 0.732 / 0.422 / 1.44 / fail; s2 0.764 / 0.379 / 1.43 / fail
- bitext_telco_category: s0 0.750 / 0.367 / 1.49 / fail; s1 0.821 / 0.255 / 1.09 / PASS; s2 0.764 / 0.285 / 1.43 / fail
- tickets_incident: s0 0.813 / 0.306 / 1.61 / fail; s1 0.800 / 0.310 / 1.60 / fail; s2 0.787 / 0.323 / 1.65 / fail
- tickets_priority: s0 0.480 / 0.727 / 5.77 / fail; s1 0.413 / 0.735 / 4.95 / fail; s2 0.467 / 0.773 / 6.41 / fail

## Verdict (2026-10-11)

**The stable recipe trains reliably (3 of 3 seeds) but passes the gate on 1 of 12 task-seed pairs. It is not better than
the no-penalty 4B run.**

- **Choice gets worse with the penalty**, consistently across seeds: bitext_category 0.750 ± 0.016 vs 0.800, telco 0.779 ± 0.038 vs
  0.800, and Brier is worse on both. The 7-point drop seen earlier was not the instability.
- **Noul is level with post-hoc** (Brier 0.313 ± 0.009 vs 0.304) but is no longer calibrated at T=1 (fitted T 1.62, against 1.16 in
  the old seed-0 run). The seed-0 calibration came with the unstable recipe and does not reproduce here.
- **Score is unchanged:** accuracy at about post-hoc's level, and still overconfident (T 5.7 ± 0.7).
- **Confound:** this recipe changes two things at once, the learning rate (1e-4 → 5e-5) and the penalty start. Less effective training could account
  for part of the Choice drop. Separating them needs no-penalty seeds at 5e-5.

**Recommendation as it stands:** at 4B, use the no-penalty `mix2` adapter with a per-family temperature: one pass, the best
accuracy, and the best Brier on 3 of 4 tasks. Drop the confidence penalty. The "no fitted T" goal is not met at 4B.
