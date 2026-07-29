# P0 — Persistence Variants

## Definition

We evaluate two deterministic persistence baselines for causal next-emotion forecasting. Both use only previously observed emotion labels and never see the target utterance.

- **P0a — Previous-turn persistence:** copy the emotion from the immediately previous turn, regardless of speaker.
- **P0b — Same-speaker persistence:** copy the target speaker’s own most recent prior emotion. If unavailable, fall back to the immediately previous turn, then the training-set majority emotion.

No text encoder, model training, or future information is used.

## Results

### IEMOCAP

| Method | Weighted F1 | Macro F1 | Accuracy | ES F1 | No-Shift F1 |
|---|---:|---:|---:|---:|---:|
| P0a — Previous turn | 0.5708 | 0.5889 | 0.5710 | 0.2853 | 0.6752 |
| P0b — Same speaker | **0.7323** | **0.7276** | **0.7337** | 0.0000 | 1.0000 |

The reproduced results confirm that the previously reported `0.7323` persistence result corresponds to the same-speaker version.

### Emotion-Prediction-Dataset

| Dataset | Method | Weighted F1 | Macro F1 | Accuracy | ES F1 | No-Shift F1 |
|---|---|---:|---:|---:|---:|---:|
| DailyDialog | P0a — Previous turn | 0.4238 | 0.1596 | 0.4320 | 0.2627 | 0.6608 |
| DailyDialog | P0b — Same speaker | **0.4436** | **0.2523** | **0.4640** | 0.0000 | 1.0000 |
| EmoryNLP | P0a — Previous turn | 0.2466 | 0.2180 | 0.2476 | 0.2094 | 0.3120 |
| EmoryNLP | P0b — Same speaker | **0.3175** | **0.2952** | **0.3193** | 0.0000 | 1.0000 |
| MELD | P0a — Previous turn | 0.3437 | 0.2419 | 0.3440 | 0.1630 | 0.6331 |
| MELD | P0b — Same speaker | **0.3843** | **0.2706** | **0.3850** | 0.0000 | 1.0000 |

## Interpretation

The same-speaker version gives the stronger overall score on every dataset, but its `0.0` ES F1 shows that it cannot predict genuine emotion shifts. It should therefore be treated as an emotional-inertia floor rather than a learned forecasting method.

## Files

- `run.py` — persistence implementation
- `results/` — saved metrics and predictions
