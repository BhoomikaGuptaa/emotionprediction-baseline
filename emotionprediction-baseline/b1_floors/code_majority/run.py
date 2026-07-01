"""
Baseline 1: Majority class
===========================
Predicts the single most frequent emotion in the TRAIN split for every test turn.
This is the trivial floor. On IEMOCAP (balanced) it should be weak; if a learned
baseline cannot beat this, something is wrong.

Usage:
  python baseline1_majority/run.py --data_path data/iemocap.pkl \
      --save_path results/b1_majority.json
"""
import os, sys, argparse
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.iemocap_utils import load_iemocap_pkl, load_iemocap_json, LABEL2ID, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--loader", default="pkl", choices=["pkl", "json"])
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--save_path", default=None)
    args = ap.parse_args()

    splits = load_iemocap_pkl(args.data_path) if args.loader == "pkl" \
        else load_iemocap_json(args.data_path)

    train = splits["train"]
    test  = splits[args.split]
    print(f"Train: {len(train)} | {args.split}: {len(test)}")

    majority = Counter(s.target_emotion for s in train).most_common(1)[0][0]
    print(f"Majority class (from train): {majority}")

    maj_id  = LABEL2ID[majority]
    y_true  = [s.target_emotion_id for s in test]
    y_pred  = [maj_id] * len(test)

    evaluate(y_true, y_pred, f"B1: Majority ({majority})", save_path=args.save_path)


if __name__ == "__main__":
    main()
