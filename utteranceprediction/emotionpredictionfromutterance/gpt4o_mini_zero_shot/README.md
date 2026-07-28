# gpt4o mini zero shot

Pipeline: dialogue history -> one predicted utterance -> fixed Qwen2.5-7B emotion labeler -> gold next-emotion comparison.

| Metric | Value |
|---|---:|
| Weighted F1 | 0.3408 |
| Macro F1 | 0.3377 |
| Accuracy | 0.3285 |
| Emotion-shift weighted F1 | 0.1754 |
| No-shift weighted F1 | 0.4051 |
| Parse failures | 4 |
