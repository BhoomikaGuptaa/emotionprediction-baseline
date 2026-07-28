# dialoguesim llama3b

Pipeline: dialogue history -> one predicted utterance -> fixed Qwen2.5-7B emotion labeler -> gold next-emotion comparison.

| Metric | Value |
|---|---:|
| Weighted F1 | 0.2931 |
| Macro F1 | 0.3055 |
| Accuracy | 0.2764 |
| Emotion-shift weighted F1 | 0.1913 |
| No-shift weighted F1 | 0.3342 |
| Parse failures | 12 |
