# Persistence and P-Series Emotion-Forecasting Baselines

This package is designed to be copied into the existing GitHub repository without reorganizing the rest of the project. It contains P0, P2, P3, P4, and P5 as self-contained folders.

## Common task

Given causal dialogue history, historical emotion labels, and the identity of the next speaker, predict the next emotion **without observing the target utterance**.

- Dataset: IEMOCAP
- Canonical dialogue split: 100 train / 20 dev / 31 test
- Canonical test points: 1,592
- Emotion labels: neutral, frustration, sadness, anger, excited, happiness

## Methods and results

| ID | Method | Category | Paper / source | Weighted F1 | Macro F1 | Accuracy | ES F1 | No-shift F1 |
|---|---|---|---|---:|---:|---:|---:|---:|
| P0 | Persistence | Deterministic inertia baseline | Standard heuristic; no single paper | 0.7323 | 0.7276 | 0.7337 |  |  |
| P2 | DAG-ERC-F | Custom causal graph adaptation | [DAG-ERC, ACL-IJCNLP 2021](https://arxiv.org/abs/2105.12907) |  |  |  |  |  |
| P3 | RoBERTa-History | Supervised history encoder | Project baseline using `roberta-base` | 0.6703 | 0.6707 | 0.6702 | 0.2160 | 0.8391 |
| P4 | PUGCN adaptation | Pseudo-utterance/graph forecasting model | [PUGCN, ESWA 2025](https://doi.org/10.1016/j.eswa.2025.127382) |  |  |  |  |  |
| P5 | Speaker transition | Deterministic train-only transition heuristic | Project-defined heuristic | 0.7267 | 0.7227 | 0.7268 | 0.0000 | 1.0000 |

Blank result cells mean that no finalized canonical test result is included yet. They are deliberately blank rather than marked as running or estimated.

## Folder guide

- `P0_Persistence/` — predicts the next emotion as the most recent observed emotion only.
- `P2_DAGERC_F/` — DAG-inspired causal graph model with a virtual target node; paper and official-code links are preserved in its README.
- `P3_RoBERTa_History/` — completed seed-42 run with code, full predictions, metadata, metrics, and selected checkpoint.
- `P4_PUGCN/` — PUGCN forecasting adaptation reconstructed from the paper; result folder is an empty placeholder.
- `P5_Speaker_Transition/` — completed deterministic heuristic with full predictions and canonical metrics.
- `results_summary.csv` — machine-readable copy of the table above.

## Important naming

- Report P2 as **DAG-ERC-F (custom causal adaptation)**, not as an exact DAG-ERC reproduction.
- Report P4 as **PUGCN adapted to canonical IEMOCAP forecasting**, not as a reproduction, because no public implementation was available in the supplied materials.
- P0 and P5 are distinct: P0 copies the last observed emotion; P5 learns target-speaker transition probabilities from the training split.
