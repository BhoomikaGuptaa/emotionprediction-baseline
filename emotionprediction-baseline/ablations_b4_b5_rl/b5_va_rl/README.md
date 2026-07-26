# B5 — Direct GRPO with Train-Derived V/A Reward

B5 predicts the next IEMOCAP emotion directly from causal dialogue history.
It uses supervised fine-tuning followed by GRPO.

## Reward

```text
0.2 × format compliance
+ 0.2 × valid-label compliance
+ 0.6 × valence/arousal similarity(predicted label, gold label)
```

The V/A component is continuous: an exact label match receives full task
credit, while affectively nearby incorrect labels receive partial credit.
There is no separate exact-match term in the task reward.

## V/A source

The six label centroids are computed from IEMOCAP's dimensional annotations
using only canonical training conversations:

1. parse `Session*/dialog/EmoEvaluation/*.txt`;
2. keep the six project emotion classes;
3. normalize original 1–5 ratings with `(rating - 3) / 2`;
4. compute one training-only centroid per emotion;
5. convert centroid distance to similarity.

This makes the reward dataset-specific, reproducible, and test-leakage-free.

## Canonical data

| Split | Conversations |
|---|---:|
| Train | 100 |
| Development | 20 |
| Test | 31 |

The test set contains 1,592 forecasting points.

## Final seed-42 results

| Stage | Weighted F1 | Macro F1 | Accuracy | ES F1 | No-shift F1 |
|---|---:|---:|---:|---:|---:|
| SFT only | 0.6304 | 0.6284 | 0.6143 | 0.1990 | 0.7874 |
| SFT → GRPO | **0.6521** | **0.6501** | **0.6413** | **0.2042** | **0.8112** |

The selected GRPO checkpoint had development weighted F1 `0.6308`.

## Commands

```bash
python run.py --mode self_test
python run.py --mode inspect --data_path /path/to/IEMOCAP_features.pkl
```

Train:

```bash
python run.py \
  --mode train \
  --data_path /path/to/IEMOCAP_features.pkl \
  --emotion_eval_dir /path/to/IEMOCAP_full_release \
  --base_model /path/to/Qwen2.5-3B-Instruct \
  --output_dir outputs/b5_full_seed42 \
  --seed 42
```

Evaluate:

```bash
python run.py \
  --mode eval \
  --data_path /path/to/IEMOCAP_features.pkl \
  --model_path outputs/b5_full_seed42/best_dev_model \
  --save_path outputs/b5_full_seed42/test_predictions.json
```

## Included artifacts

- implementation and shared loader;
- train-only V/A centroids and similarity metadata;
- development checkpoint scores;
- SFT-only and GRPO predictions;
- final metrics, run metadata, and logs.

## Excluded artifacts

- IEMOCAP data;
- base-model weights;
- adapters/checkpoints;
- container images and caches.

This is a single seed-42 result, not a multi-seed estimate.
