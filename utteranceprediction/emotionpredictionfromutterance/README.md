# Emotion prediction from one predicted utterance

These experiments test a strict utterance bottleneck:

`dialogue history -> generate one next utterance -> label that utterance -> compare with gold next emotion`

The final emotion labeler does not receive the original history. This differs from the candidate-reasoning experiments in `../multipleutteranceprediction/`, where history remains available to the final predictor.

| Utterance source | Weighted F1 | ES F1 | No-shift F1 |
|---|---:|---:|---:|
| Qwen2.5-7B zero-shot | 0.4488 | 0.2398 | 0.5284 |
| Parrot control | 0.4016 | 0.1727 | 0.4906 |
| GPT-4o-mini | 0.3408 | 0.1754 | 0.4051 |
| Dialogue-Sim Qwen2.5-7B | 0.3136 | 0.2217 | 0.3542 |
| Dialogue-Sim SFT-only | 0.2935 | 0.1730 | 0.3414 |
| Dialogue-Sim Llama3B | 0.2931 | 0.1913 | 0.3342 |
