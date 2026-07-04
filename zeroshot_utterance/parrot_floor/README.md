# Parrot Floor

No model. The "prediction" for the next utterance is a word-for-word copy of the previous utterance:

    prediction = history[-1]

Zero GPU, runs in seconds. See the top-level README for why this exists and what it means.

Run:
```bash
python generate_nextutt.py --pkl IEMOCAP_features.pkl --parrot_floor \
  --labels_in_history 1 --out parrot_floor.json --preds_out parrot_preds.jsonl
```

Result files: `parrot_floor.json`, `parrot_preds.jsonl`
