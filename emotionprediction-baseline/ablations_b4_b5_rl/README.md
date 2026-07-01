# Ablations - B4 / B5 (Reinforcement Learning, reward-design ablation)

These are NOT external baselines. They are two variants of our own SFT -> GRPO pipeline that differ ONLY in the reward, included as a reward-design ablation.

**Method:** Qwen2.5-3B + LoRA. Stage 1: SFT (2 epochs) to teach the output format. Stage 2: GRPO reinforcement learning (300 steps). The two variants share the SFT stage and diverge only at the reward.

**B4 - discrete reward:** 0.2 format + 0.2 valid-label + 0.6 exact-match correct.

**B5 - V/A reward:** 0.2 format + 0.2 valid-label + 0.6 x valence/arousal similarity (partial credit for close emotions).

**Task:** Emotion forecasting (causal next-emotion prediction): given the dialogue history (past turns with their gold emotion labels) and the next speaker, predict the next turn's emotion WITHOUT seeing that turn's text. Metric: prior weighted F1 over turns t>=1.

**Settings:** Qwen2.5-3B + LoRA (r=16, alpha=32). SFT 2 epochs. GRPO 300 steps, 2 candidates/prompt. Seeds 42 and 43 (2-seed mean). The target emotion is used ONLY inside the reward function, never in the model's prompt (no leakage; no retrieval mechanism exists in this pipeline).

## Results (IEMOCAP, prior weighted F1, t>=1)

| Variant | Stage | Weighted F1 | Macro F1 | Accuracy | Parse-fail |
| --- | --- | --- | --- | --- | --- |
| B4 discrete | SFT only (s43) | 0.6129 | | 0.5948 | 5.65% |
| B4 discrete | GRPO (s43) | 0.6295 | | 0.6187 | 3.77% |
| B4 discrete | GRPO (2-seed mean) | 0.6397 | | | |
| B5 V/A | SFT only (s43) | 0.6129 | | 0.5948 | 5.65% |
| B5 V/A | GRPO (s43) | 0.6324 | | 0.6256 | 2.32% |
| B5 V/A | GRPO (2-seed mean) | 0.6361 | | | |

Note: B4 and B5 share the SFT stage (identical SFT numbers are expected, not a copy-paste error). GRPO improves over SFT in every run. On exact-match the two rewards tie; the V/A reward's advantage appears only under a continuous metric.
