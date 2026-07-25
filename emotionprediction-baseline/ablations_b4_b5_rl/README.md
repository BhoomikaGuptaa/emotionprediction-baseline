# B4 / B5 Reward Ablations

B4 and B5 are two variants of the same SFT → GRPO emotion-forecasting pipeline. They differ only in the GRPO reward.

## Task

Given the dialogue history, previous emotion labels, and the next speaker, predict the next emotion without seeing the target utterance text.

## Setup

- Base model: Qwen2.5-3B-Instruct
- LoRA: `r=16`, `alpha=32`
- SFT: 2 epochs
- GRPO: 300 steps
- Candidates per prompt: 2
- Seed: 42
- Dataset: canonical IEMOCAP 100/20/31 split
- Primary metric: prior weighted F1 over targets at `t >= 1`

## Rewards

### B4: Discrete reward

- 0.2 for correct format
- 0.2 for a valid emotion label
- 0.6 for an exact match

### B5: Valence/Arousal reward

- 0.2 for correct format
- 0.2 for a valid emotion label
- 0.6 × valence/arousal similarity

## Results

| Variant | Stage | Weighted F1 | Macro F1 | Accuracy | Parse Failures | ES F1 | No-Shift F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| B4 discrete | SFT only | **0.6304** | **0.6284** | **0.6143** | **4.59%** | **0.1990** | **0.7874** |
| B4 discrete | SFT → GRPO | **0.6320** | **0.6304** | **0.6225** | **2.9%** | **0.1967** | **0.7876** |
| B5 V/A | SFT only | — | — | — | — | — | — |
| B5 V/A | SFT → GRPO | — | — | — | — | — | — |

For B4, the best development checkpoint was selected at step 200 with development weighted F1 `0.6328`.

## Status

- B4 SFT-only: running
- B4 SFT → GRPO: complete
- B5 SFT-only: pending
- B5 SFT → GRPO: pending
