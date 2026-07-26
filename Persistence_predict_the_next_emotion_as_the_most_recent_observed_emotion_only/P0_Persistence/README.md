# P0 — Persistence

## Definition

Predict the next emotion as the **most recent observed emotion** in the causal dialogue history. The target utterance is never observed.

This is a deterministic, label-only emotional-inertia baseline. It uses no text encoder, no model training, and no future information.

## Original source

- **Paper:** No single paper; this is a standard heuristic baseline.
- **Original code:** Not applicable.
- **Task adaptation:** Applied to causal next-emotion forecasting on the canonical IEMOCAP split.

## Result

| Weighted F1 | Macro F1 | Accuracy |
|---:|---:|---:|
| 0.7323 | 0.7276 | 0.7337 |

The values above are preserved from the existing repository README. The uploaded snapshot did not include a standalone P0 prediction file, so `results/metrics.json` records the reported result and its provenance.

## Files

- `run.py` — persistence baseline implementation from the existing repository.
- `shared/` — loader used by the original baseline folder.
- `results/metrics.json` — machine-readable reported metrics.
