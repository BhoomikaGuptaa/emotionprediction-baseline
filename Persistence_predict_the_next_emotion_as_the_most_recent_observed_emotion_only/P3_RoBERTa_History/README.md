# P3 — RoBERTa-History

## What it does

Fine-tunes `roberta-base` to predict the next emotion directly from causal dialogue history. It uses up to ten previous turns and never receives the target utterance.

## Method status

This is a project baseline rather than a reproduction of a single published architecture. The implementation, configuration, selected checkpoint, complete test predictions, and seed-42 metadata are included.

## Configuration

- Encoder: `roberta-base`
- Maximum length: 256
- Maximum history turns: 10
- Learning rate: `2e-5`
- Batch size: 8
- Gradient accumulation: 2
- Epochs: 5
- Weight decay: 0.01
- Seed: 42
- Best checkpoint selected by development weighted F1

## Canonical test result

| Weighted F1 | Macro F1 | Accuracy | ES F1 | No-shift F1 | Test points |
|---:|---:|---:|---:|---:|---:|
| 0.6703 | 0.6707 | 0.6702 | 0.2160 | 0.8391 | 1,592 |

## Files

- `run.py`, `config.yaml`, notebooks, and smoke test — executable implementation.
- `results/seed42/predictions.json` — complete rich-format predictions and metrics.
- `results/seed42/run_metadata.json` — resolved configuration and selected dev score.
- `results/seed42/best_model/` — saved selected checkpoint.
