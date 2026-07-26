# P2 DAG-ERC-F Final Results

## Final test metrics

| Metric | Score |
|---|---:|
| Weighted F1 | **0.6922** |
| Macro F1 | 0.6885 |
| Accuracy | 0.6954 |
| Parse failures | 0 / 1,592 |

## Per-class F1

| Emotion | F1 |
|---|---:|
| Neutral | 0.6796 |
| Frustration | 0.6255 |
| Sadness | 0.7977 |
| Anger | 0.5648 |
| Excited | 0.7967 |
| Happiness | 0.6667 |

## Shift breakdown

| Subset | Count | Weighted F1 |
|---|---:|---:|
| Emotion shift | 410 | 0.2956 |
| No shift | 1,151 | 0.8386 |
| Undefined | 31 | Excluded |

The shift definition compares the target emotion with the most recent earlier labelled turn from the same speaker.

## Development selection

- Best development weighted F1: `0.6624`
- Selected epoch: `6`
- Training epochs completed: `30`
- Seed: `42`

## Analysis

P2 is the strongest learned forecasting baseline among P2, P3, P4, B4, and B5 in the current single-seed table, with weighted F1 `0.6922`. It remains below persistence (`0.7323`) and speaker-aware persistence (`0.7267`).

The model is highly effective on no-shift cases (`0.8386`) but much weaker at same-speaker emotion shifts (`0.2956`). This indicates that most of its aggregate strength comes from modeling conversational continuity rather than reliably anticipating state changes.

Compared with P4, P2 gains substantially overall (`0.6922` versus `0.5186`) and on no-shift cases, while P4 has the stronger shift score (`0.4138` versus `0.2956`). This suggests that P4's generated pseudo-utterance path may add some shift sensitivity, but at a large cost to ordinary-case accuracy.

This is a single-seed result and should not be represented as a multi-seed estimate.
