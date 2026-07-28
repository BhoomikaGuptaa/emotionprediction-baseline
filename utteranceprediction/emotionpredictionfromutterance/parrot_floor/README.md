# parrot floor

Pipeline: dialogue history -> one predicted utterance -> fixed Qwen2.5-7B emotion labeler -> gold next-emotion comparison.

| Metric | Value |
|---|---:|
| Weighted F1 | 0.4016 |
| Macro F1 | 0.4128 |
| Accuracy | 0.3894 |
| Emotion-shift weighted F1 | 0.1727 |
| No-shift weighted F1 | 0.4906 |
| Parse failures | 3 |
