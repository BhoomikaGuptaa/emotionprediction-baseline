# Emotion Prediction Baselines

Baselines for **causal next-emotion prediction** (emotion forecasting) in conversation.

**Task:** given the dialogue history (past turns with their gold emotion labels) and the next speaker, predict the next turn's emotion **without seeing that turn's text**. Metric: **prior weighted F1** over turns t>=1. Primary dataset: IEMOCAP (test = 31 dialogues, 1592 forecasting points).

Each baseline is a faithful reimplementation following its paper, adapted to forecasting by changing only the data (recognition -> prediction). Folders b4/b5 are our own reward-design ablation, not external baselines.

## Summary (IEMOCAP, prior weighted F1)

| # | Baseline | Family | Weighted F1 |
| --- | --- | --- | --- |
| B1 | Majority floor | heuristic | 0.092 |
| B1 | Persistence (the bar) | heuristic | 0.7323 |
| B1 | Transition matrix | heuristic | 0.5708 |
| B2 | Zero-shot (Llama-8B) | prompting | 0.5981 |
| B3 | Few-shot k=6 (Qwen-3B) | prompting | 0.6223 |
| B6 | InstructERC | generative LLM + retrieval |0.6678 |
| B7 | EACL | contrastive encoder | 0.5921 |
| B8 | DialogueRNN | recurrent | 0.4714 |
| B9 | DualGATs | graph (planned) | |
| -- | B4 discrete RL (ablation) | RL | 0.6397 |
| -- | B5 V/A RL (ablation) | RL | 0.6361 |

Empty cells = results to be confirmed / runs pending.

## Folders

- `b1_floors/` - majority, persistence, transition matrix
- `b2_zeroshot/` - zero-shot LLM
- `b3_fewshot/` - few-shot LLM (k=6)
- `b6_instructerc/` - InstructERC (arXiv 2023)
- `b7_eacl/` - EACL (Findings of NAACL 2024)
- `b8_dialoguernn/` - DialogueRNN (AAAI 2019)
- `b9_dualgats/` - DualGATs (ACL 2023, planned)
- `ablations_b4_b5_rl/` - our RL reward-design ablation (not a baseline)
