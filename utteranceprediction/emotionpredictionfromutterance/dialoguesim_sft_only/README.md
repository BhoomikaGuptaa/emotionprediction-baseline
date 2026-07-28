# dialoguesim sft only

Pipeline: dialogue history -> one predicted utterance -> fixed Qwen2.5-7B emotion labeler -> gold next-emotion comparison.

| Metric | Value |
|---|---:|
| Weighted F1 | 0.2935 |
| Macro F1 | 0.2924 |
| Accuracy | 0.2852 |
| Emotion-shift weighted F1 | 0.1730 |
| No-shift weighted F1 | 0.3414 |
| Parse failures | 7 |
