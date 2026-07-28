# qwen2.5 7b zero shot

Pipeline: dialogue history -> one predicted utterance -> fixed Qwen2.5-7B emotion labeler -> gold next-emotion comparison.

| Metric | Value |
|---|---:|
| Weighted F1 | 0.4488 |
| Macro F1 | 0.4352 |
| Accuracy | 0.4504 |
| Emotion-shift weighted F1 | 0.2398 |
| No-shift weighted F1 | 0.5284 |
| Parse failures | 56 |
