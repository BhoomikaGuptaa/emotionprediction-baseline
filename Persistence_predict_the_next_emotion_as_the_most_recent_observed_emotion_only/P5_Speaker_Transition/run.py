"""
P5 — Speaker transition (CPU-only heuristic)
==============================================
Estimates P(target emotion | target speaker's OWN most recent previous
emotion), learned as a Laplace-smoothed transition matrix from the TRAIN
split only. Three-tier fallback, used in this exact order per sample:

  1. LEARNED SAME-SPEAKER TRANSITION -- if the target speaker has a prior
     turn somewhere in this sample's history, predict
     argmax_e P_smoothed(e | that speaker's most recent own emotion).
  2. GLOBAL MAJORITY -- if the target speaker has no prior labelled turn.

There is deliberately no any-speaker fallback: another speaker's emotion is not the target speaker's emotional state.

Note the difference from B1's persistence baseline elsewhere in this
project: that baseline predicts a straight COPY of the same-speaker prior
emotion. This heuristic instead predicts the LEARNED most-likely NEXT
emotion given that prior emotion (a real, if simple, transition model,
using counts collected only from train) -- tier 1 here is strictly more
informed than plain persistence, and only degrades to persistence-like
behavior in tier 2 when there isn't enough same-speaker history to learn
from.

Run:
  python run.py --data_path /path/to/iemocap.pkl --config config.yaml \
      --save_path outputs/predictions.json
"""
import os, sys, argparse, random, json, subprocess, datetime
from collections import Counter, defaultdict
import numpy as np
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from shared.iemocap_utils import load_iemocap_pkl, EMOTION_LABELS, LABEL2ID, NUM_LABELS, evaluate


def load_config(path, overrides):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def git_commit_or_na():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "n/a"


def strict_same_speaker_prev(sample):
    """The target speaker's own most recent prior emotion, STRICT -- unlike
    shared.iemocap_utils.prev_speaker_emotion, this does NOT fall back to
    any-speaker if the target speaker has no prior turn. Returns None in
    that case (tier 1 is not applicable; caller falls through to tier 2)."""
    spk = sample.target_speaker
    for i in range(len(sample.history) - 1, -1, -1):
        if sample.history_speakers[i] == spk and sample.history_emotions[i] is not None:
            return sample.history_emotions[i]
    return None


def build_transition_matrix(train_samples, alpha):
    """P_smoothed(next=e_j | prev=e_i), Laplace-smoothed, learned ONLY from
    train samples where strict_same_speaker_prev is defined (tier-1
    applicable samples). Returns dict: prev_label -> list[NUM_LABELS] probs."""
    counts = {e: [0] * NUM_LABELS for e in EMOTION_LABELS}
    n_tier1_train = 0
    for s in train_samples:
        prev = strict_same_speaker_prev(s)
        if prev is not None and prev in counts:
            counts[prev][s.target_emotion_id] += 1
            n_tier1_train += 1

    smoothed = {}
    for prev_e, row in counts.items():
        total = sum(row) + alpha * NUM_LABELS
        smoothed[prev_e] = [(c + alpha) / total for c in row]
    return smoothed, n_tier1_train


def predict(sample, transition, majority_id):
    prev = strict_same_speaker_prev(sample)
    if prev is not None and prev in transition:
        return int(np.argmax(transition[prev])), "tier1_learned_transition"
    return majority_id, "tier2_majority_no_same_speaker_history"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--save_path", default="outputs/predictions.json")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, {"alpha": args.alpha, "seed": args.seed})
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"])

    splits = load_iemocap_pkl(args.data_path)
    train_s = splits["train"]
    majority_label = Counter(s.target_emotion for s in train_s).most_common(1)[0][0]
    majority_id = LABEL2ID[majority_label]

    transition, n_tier1_train = build_transition_matrix(train_s, cfg["alpha"])
    print(f"Learned transition matrix from {n_tier1_train}/{len(train_s)} train samples "
          f"with a same-speaker prior emotion (alpha={cfg['alpha']}).")
    print("Learned P(next | prev) argmax per prev emotion:")
    for e in EMOTION_LABELS:
        row = transition[e]
        best = EMOTION_LABELS[int(np.argmax(row))]
        print(f"  prev={e:<12} -> argmax next={best}  (P={max(row):.3f})")

    test = splits[args.split]
    y_true, y_pred, tiers = [], [], []
    for s in test:
        pred_id, tier = predict(s, transition, majority_id)
        y_true.append(s.target_emotion_id); y_pred.append(pred_id); tiers.append(tier)

    tier_counts = Counter(tiers)
    print(f"\nFallback tier usage on {args.split}: {dict(tier_counts)}")

    results = evaluate(y_true, y_pred, "P5 Speaker transition (Laplace-smoothed, majority fallback)",
                       save_path=args.save_path, samples=test)

    # Also fold tier usage into a metadata file alongside the predictions.
    meta_path = os.path.join(os.path.dirname(args.save_path) or ".", "run_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({"command": " ".join(sys.argv), "config": cfg,
                   "n_tier1_train_samples": n_tier1_train,
                   "tier_usage_on_eval_split": dict(tier_counts),
                   "weighted_f1": results["weighted_f1"],
                   "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
                   "git_commit": git_commit_or_na()}, f, indent=2)
    print(f"Run metadata -> {meta_path}")


if __name__ == "__main__":
    main()
