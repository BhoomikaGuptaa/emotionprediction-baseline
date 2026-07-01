"""
Baseline: First-order emotion transition matrix
===============================================
Learns P(y_t | y_{t-1}) from TRAIN (y_{t-1} = the immediately preceding turn's
emotion) and predicts argmax at test time. A pure Markov floor: it captures the
emotion-dynamics structure (which DPM's joint-transition head is built around)
without reading any text.

Persistence vs. this transition baseline is itself informative: if the transition
matrix beats persistence, transitions are not purely self-loops; if it does not,
inertia dominates. Either way it bounds how much the text models actually add.

Usage:
  python baseline_transition/run.py --data_path data/iemocap.pkl \
      --save_path results/b_transition.json
"""
import os, sys, argparse
from collections import Counter, defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.iemocap_utils import (
    load_iemocap_pkl, load_iemocap_json, EMOTION_LABELS, LABEL2ID, evaluate,
)


def prev_emotion(sample):
    for i in range(len(sample.history_emotions) - 1, -1, -1):
        if sample.history_emotions[i] is not None:
            return sample.history_emotions[i]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--loader", default="pkl", choices=["pkl", "json"])
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--save_path", default=None)
    args = ap.parse_args()

    splits = load_iemocap_pkl(args.data_path) if args.loader == "pkl" \
        else load_iemocap_json(args.data_path)

    # Learn transition counts on TRAIN: prev_emotion -> argmax next emotion.
    trans = defaultdict(Counter)
    for s in splits["train"]:
        pe = prev_emotion(s)
        trans[pe][s.target_emotion] += 1
    best_next = {pe: c.most_common(1)[0][0] for pe, c in trans.items()}
    majority = Counter(s.target_emotion for s in splits["train"]).most_common(1)[0][0]

    print("Learned transitions (prev -> most likely next):")
    for e in [None] + EMOTION_LABELS:
        if e in best_next:
            print(f"  {str(e):<12} -> {best_next[e]}")

    test = splits[args.split]
    y_true = [s.target_emotion_id for s in test]
    y_pred = [LABEL2ID[best_next.get(prev_emotion(s), majority)] for s in test]
    evaluate(y_true, y_pred, "First-order transition matrix",
             save_path=args.save_path)


if __name__ == "__main__":
    main()
