# B1 - Floors (majority / persistence / transition)

**Paper:** No paper (standard heuristic baselines).

**Paper link:** -

**Original code:** -

**Year / Conference:** -

**Original task:** Heuristic floors, no training. Standard reference points used across ERC/forecasting.

**What I did:** Adapted the standard floors to the forecasting task: majority class, persistence (repeat the last turn's emotion), and a first-order transition matrix learned from training data. Label-only, no text, no training.

**Settings:** No training. Rules computed directly on the data.

## Results (IEMOCAP, prior weighted F1, t>=1, no x_t)

| Method | Weighted F1 | Macro F1 | Accuracy |
| --- | --- | --- | --- |
| Majority class | 0.092 | 0.06 | 0.24 |
| Persistence (repeat last) | 0.7323 | 0.7276 | 0.7337 |
| Transition matrix | 0.5708 | 0.5889 | 0.5710 |

Persistence at 0.7323 is the bar every learned model must beat.
