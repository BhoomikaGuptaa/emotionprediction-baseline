# Canonical IEMOCAP Forecasting: B5, P2, and P4

This trimmed repository keeps only the three finalized methods requested:

- `B5` — direct GRPO with train-only valence/arousal similarity reward
- `P2` — causal DAG-ERC forecasting adaptation
- `P4` — PUGCN-style pseudo-utterance forecasting adaptation

## Final seed-42 results

| Method | Weighted F1 | Macro F1 | Accuracy | ES F1 | No-shift F1 |
|---|---:|---:|---:|---:|---:|
| P2 DAG-ERC-F | 0.6922 | 0.6885 | 0.6954 | 0.2956 | 0.8386 |
| B5 V/A GRPO | 0.6521 | 0.6501 | 0.6413 | 0.2042 | 0.8112 |
| P4 PUGCN | 0.5186 | 0.5190 | 0.5226 | 0.4138 | 0.5585 |

All results use the canonical 100/20/31 conversation split and 1,592 test forecasting points.

## Layout

```text
Persistence_predict_the_next_emotion_as_the_most_recent_observed_emotion_only/
  P2_DAGERC_F/
  P4_PUGCN/

emotionprediction-baseline/
  ablations_b4_b5_rl/
    b5_va_rl/
```
