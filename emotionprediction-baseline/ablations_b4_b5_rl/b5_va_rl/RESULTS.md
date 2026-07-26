# B5 Final Results — Seed 42

B5 is the direct SFT→GRPO baseline whose task reward is continuous valence/arousal similarity derived from canonical IEMOCAP training annotations.

## Reward

```text
0.2 × strict format
+ 0.2 × valid label
+ 0.6 × V/A similarity(predicted label, gold label)
```

The V/A centroids are computed from training dialogues only. The final B5 implementation does not use a separate exact-match reward term.

## Test results

| Stage | Weighted F1 | Macro F1 | Accuracy | Parse failures | ES F1 | No-shift F1 |
|---|---:|---:|---:|---:|---:|---:|
| SFT only | 0.6304 | 0.6284 | 0.6143 | 73/1592 | 0.1990 | 0.7874 |
| SFT→GRPO | **0.6521** | **0.6501** | **0.6413** | 47/1592 | **0.2042** | **0.8112** |

Canonical test size: 1,592 forecasting points. Shift evaluation uses the same-speaker definition: 410 shift, 1,151 no-shift, and 31 undefined cases.

## Interpretation

GRPO improves weighted F1 by 0.0217 over the SFT-only checkpoint. Most of the improvement comes from no-shift cases, while shift performance improves only slightly. This is a single-seed controlled result and should not be treated as proof that the reward change alone will generalize across seeds.
