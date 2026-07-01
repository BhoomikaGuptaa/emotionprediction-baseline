"""
Baseline: Emotional persistence (inertia)
=========================================
Predicts y_t = the most recent emotion of the SAME speaker before turn t
(falls back to the previous turn's emotion, then to train-majority).

This is the single most important floor for a FORECASTING paper. Emotions are
strongly autocorrelated within a speaker (this is DPM's central appraisal claim),
so "just repeat the speaker's last emotion" is a hard, near-zero-cost baseline.
If a learned model cannot beat persistence, it has not learned anything beyond
inertia. Costs nothing to run and is highly diagnostic.

Usage:
  python baseline_persistence/run.py --data_path data/iemocap.pkl \
      --save_path results/b_persistence.json
"""
import os, sys, argparse
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.iemocap_utils import load_iemocap_pkl, load_iemocap_json, LABEL2ID, evaluate


def predict(sample, majority_id):
    spk = sample.target_speaker
    # most recent SAME-speaker emotion
    for i in range(len(sample.history) - 1, -1, -1):
        if sample.history_speakers[i] == spk and sample.history_emotions[i] is not None:
            return LABEL2ID[sample.history_emotions[i]]
    # fallback: previous turn's emotion (any speaker)
    for i in range(len(sample.history_emotions) - 1, -1, -1):
        if sample.history_emotions[i] is not None:
            return LABEL2ID[sample.history_emotions[i]]
    return majority_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--loader", default="pkl", choices=["pkl", "json"])
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--save_path", default=None)
    args = ap.parse_args()

    splits = load_iemocap_pkl(args.data_path) if args.loader == "pkl" \
        else load_iemocap_json(args.data_path)
    majority = Counter(s.target_emotion for s in splits["train"]).most_common(1)[0][0]
    maj_id = LABEL2ID[majority]

    test = splits[args.split]
    y_true = [s.target_emotion_id for s in test]
    y_pred = [predict(s, maj_id) for s in test]
    evaluate(y_true, y_pred, "Persistence (same-speaker last emotion)",
             save_path=args.save_path)


if __name__ == "__main__":
    main()
