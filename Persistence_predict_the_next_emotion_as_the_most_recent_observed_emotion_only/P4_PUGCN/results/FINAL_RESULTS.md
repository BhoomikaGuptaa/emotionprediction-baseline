# P4 PUGCN Final Results

## Task

Canonical IEMOCAP next-emotion forecasting.

Given dialogue history through turn `t-1`, historical speaker identities,
historical gold emotion labels, and the identity of the next speaker,
predict the emotion at turn `t` without observing the target utterance.

## Canonical split

| Split | Conversations |
|---|---:|
| Train | 100 |
| Development | 20 |
| Test | 31 |

The canonical test set contains 1,592 forecasting points.

## Final configuration

- Method: P4 PUGCN forecasting adaptation
- Encoder/generator: `facebook/bart-base`
- Seed: 42
- Epochs: 15
- Training points: 4,730
- Development points: 960
- Test points: 1,592
- Checkpoint selected using development weighted F1
- Best development weighted F1: 0.5245
- Best epoch: 15

## Final test results

| Metric | Score |
|---|---:|
| Weighted F1 | 0.5186 |
| Macro F1 | 0.5190 |
| Accuracy | 0.5226 |
| Parse failures | 0 / 1,592 |

## Emotion-shift evaluation

| Subset | Count | Weighted F1 |
|---|---:|---:|
| Emotion shift | 410 | 0.4138 |
| No shift | 1,151 | 0.5585 |
| Undefined | 31 | Excluded |

Undefined cases have no previous labelled turn for the same speaker.

## Per-class F1

| Emotion | F1 |
|---|---:|
| Neutral | 0.3754 |
| Frustration | 0.5723 |
| Sadness | 0.5372 |
| Anger | 0.5183 |
| Excited | 0.6421 |
| Happiness | 0.4687 |

## Notes

This is a single seed-42 result and should not be presented as a multi-seed estimate.
