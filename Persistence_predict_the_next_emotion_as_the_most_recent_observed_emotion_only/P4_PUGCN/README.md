# P4 — PUGCN-Style Forecasting Adaptation

P4 is a PUGCN-style adaptation for canonical IEMOCAP next-emotion forecasting.
It is an adaptation, not an exact reproduction of the original PUGCN system.

## Forecasting task

Given dialogue history through turn `t-1`, historical speaker identities,
historical gold emotion labels, and the known next-speaker identity, predict
the emotion at turn `t` without observing the target utterance.

## Model

The implementation uses BART to:

1. encode causal dialogue history;
2. generate a pseudo target utterance from history and next-speaker context;
3. encode the generated pseudo utterance;
4. combine dialogue and pseudo-utterance representations;
5. predict one of the six IEMOCAP emotion labels.

Training combines classification loss, pseudo-utterance generation loss, and
supervised contrastive loss. The real target utterance is used only as
training supervision for the generation loss and is not passed into the
classification path. `smoke_test.py` checks this separation.

## Canonical data

| Split | Conversations | Forecasting points |
|---|---:|---:|
| Train | 100 | 4,730 |
| Development | 20 | 960 |
| Test | 31 | 1,592 |

## Final seed-42 result

| Metric | Score |
|---|---:|
| Weighted F1 | **0.5186** |
| Macro F1 | 0.5190 |
| Accuracy | 0.5226 |
| Emotion-shift weighted F1 | 0.4138 |
| No-shift weighted F1 | 0.5585 |
| Parse failures | 0 / 1,592 |

Best development weighted F1: `0.5245` at epoch 15.

## Reproduction

Install dependencies:

```bash
pip install -r requirements.txt
```

For an offline cluster, download `facebook/bart-base` in advance and set the
`encoder` value in `config.yaml` to the local directory. For an online run,
leave it as `facebook/bart-base`.

Train:

```bash
python run.py \
  --mode train \
  --data_path /path/to/IEMOCAP_features.pkl \
  --config config.yaml \
  --output_dir outputs/full_seed42 \
  --seed 42
```

Evaluate:

```bash
python run.py \
  --mode eval \
  --data_path /path/to/IEMOCAP_features.pkl \
  --config config.yaml \
  --model_path outputs/full_seed42/best.pt \
  --split test \
  --save_path outputs/full_seed42/predictions.json
```

## Included artifacts

- implementation and configuration;
- leakage/architecture smoke tests;
- final predictions for all 1,592 test points;
- run metadata;
- final training and evaluation logs;
- complete metric summary under `results/FINAL_RESULTS.md`.

## Excluded artifacts

- IEMOCAP data;
- BART weights;
- trained checkpoint;
- container image and caches.

This is a single seed-42 result, not a multi-seed estimate.
