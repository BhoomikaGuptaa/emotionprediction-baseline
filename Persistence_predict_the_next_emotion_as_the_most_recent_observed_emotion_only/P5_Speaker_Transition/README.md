# P5 — Speaker-Transition Heuristic

## What it does

Estimates a Laplace-smoothed transition distribution from the target speaker's most recent previous emotion using **training-split counts only**. If the target speaker has no prior labeled turn, it falls back to the training-set majority class.

This is different from P0: P0 copies the most recent observed emotion, while P5 learns the most likely next emotion conditioned on the target speaker's own previous emotion.

## Original source

- **Paper:** No single paper; this is a project-defined deterministic heuristic.
- **Original code:** Not applicable.
- **Task adaptation:** Built specifically for causal emotion forecasting on IEMOCAP.

## Canonical test result

| Weighted F1 | Macro F1 | Accuracy | ES F1 | No-shift F1 | Test points |
|---:|---:|---:|---:|---:|---:|
| 0.7267 | 0.7227 | 0.7268 | 0.0000 | 1.0000 | 1,592 |

The shift breakdown shows that P5 is an inertia-heavy heuristic: it is perfect on no-shift cases and fails on all true same-speaker emotion shifts.

## Files

- `run.py`, `config.yaml`, notebook, and smoke test — executable implementation.
- `results/seed42/predictions.json` — complete canonical test predictions.
- `results/seed42/metrics.json` — full canonical evaluation, including per-class and shift metrics.
- `results/seed42/run_metadata.json` — resolved configuration and tier usage.
