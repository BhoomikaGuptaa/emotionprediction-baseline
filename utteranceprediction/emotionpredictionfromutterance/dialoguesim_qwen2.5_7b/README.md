# dialoguesim qwen2.5 7b

Pipeline: dialogue history -> one predicted utterance -> fixed Qwen2.5-7B emotion labeler -> gold next-emotion comparison.

| Metric | Value |
|---|---:|
| Weighted F1 | 0.3136 |
| Macro F1 | 0.3217 |
| Accuracy | 0.3015 |
| Emotion-shift weighted F1 | 0.2217 |
| No-shift weighted F1 | 0.3542 |
| Parse failures | 5 |
