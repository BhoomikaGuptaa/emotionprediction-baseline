# Matched one-candidate versus four-candidate ablation

Model for candidate generation and final emotion prediction: `Qwen/Qwen2.5-7B-Instruct`.

The notebook generates one shared set of four spoken-text-only candidates per test example and evaluates three arms:

- **B-like:** candidate 1 only -> emotion. The final predictor does not receive history.
- **C1:** history + candidate 1 -> emotion.
- **C4:** history + all four candidates -> emotion.

The main controlled comparison is **C1 versus C4**, because both retain the same history and differ only in candidate count.

## Results

| Arm | Weighted F1 | Macro F1 | Accuracy | ES F1 | No-shift F1 | Parse failures |
|---|---:|---:|---:|---:|---:|---:|
| B-like | pending | pending | pending | pending | pending | pending |
| C1: history + 1 candidate | pending | pending | pending | pending | pending | pending |
| C4: history + 4 candidates | pending | pending | pending | pending | pending | pending |

Replace the pending values after the notebook finishes; no other repository folders need to change.
