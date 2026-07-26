# P2 — DAG-ERC-F Custom Causal Adaptation

P2 is a custom DAG-inspired causal graph baseline for next-emotion forecasting on canonical IEMOCAP.

It receives dialogue history through turn `t-1`, historical speakers and labels, and the known next-speaker identity. It predicts the emotion at turn `t` without observing the target utterance.

## Important naming

Use one of these names in reports:

- `DAG-ERC-F (custom causal adaptation)`
- `DAG-inspired causal graph baseline`

Do not describe this as an exact reproduction of the original DAG-ERC paper. The original method performs conversational emotion recognition using observed utterances; this adaptation inserts a virtual target node with no target text.

## Architecture

- Frozen `roberta-large` history features
- Directed acyclic local graph over historical turns
- Same-speaker versus different-speaker edge typing
- Learned virtual target node
- Four graph layers
- Target-speaker identity used for target-node edge typing
- Checkpoint selected by development weighted F1

## Canonical data

| Split | Conversations | Forecasting points |
|---|---:|---:|
| Train | 100 | 4,730 |
| Development | 20 | 960 |
| Test | 31 | 1,592 |

## Final seed-42 result

| Metric | Score |
|---|---:|
| Weighted F1 | **0.6922** |
| Macro F1 | 0.6885 |
| Accuracy | 0.6954 |
| Parse failures | 0 / 1,592 |
| Emotion-shift weighted F1 | 0.2956 |
| No-shift weighted F1 | 0.8386 |

Best development weighted F1: `0.6624` at epoch 6.

See `results/FINAL_RESULTS.md` for the full class breakdown and interpretation.

## Reproduction

Install dependencies:

```bash
pip install -r requirements.txt
```

Create the frozen feature cache:

```bash
python run.py \
  --mode cache \
  --data_path /path/to/IEMOCAP_features.pkl \
  --config config.yaml \
  --cache_dir outputs/cache
```

Train:

```bash
python run.py \
  --mode train \
  --data_path /path/to/IEMOCAP_features.pkl \
  --config config.yaml \
  --cache_dir outputs/cache \
  --output_dir outputs/full
```

Evaluate:

```bash
python run.py \
  --mode eval \
  --data_path /path/to/IEMOCAP_features.pkl \
  --config config.yaml \
  --cache_dir outputs/cache \
  --model_path outputs/full/best.pt \
  --split test \
  --save_path outputs/full/predictions.json
```

A Nautilus notebook is included. The checkpoint filename produced by `run.py` is `best.pt`.

## Files intentionally excluded

- IEMOCAP data
- frozen feature cache
- trained `best.pt` checkpoint
- Python caches

The final predictions and run metadata are included under `results/`.
