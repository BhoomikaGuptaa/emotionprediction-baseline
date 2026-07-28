# Original four-candidate reasoning

Pipeline:

`history -> generate four possible next utterances -> history + all four candidates -> one emotion label`

The original generation run was preserved and reparsed without regenerating candidates.

| Metric | Value |
|---|---:|
| Weighted F1 | 0.6271 |
| Macro F1 | 0.6257 |
| Accuracy | 0.6237 |
| Emotion-shift weighted F1 | 0.1567 |
| No-shift weighted F1 | 0.7961 |
| Parse failures | 38 |
| Parser repairs | 345 |

This result is retained as an earlier experiment. The matched C1/C4 notebook removes explicit emotion-tag formatting and provides the controlled one-versus-four comparison.
