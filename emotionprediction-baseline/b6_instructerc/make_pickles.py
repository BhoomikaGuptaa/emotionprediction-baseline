#!/usr/bin/env python3
"""
Convert MELD (CSV) and EmoryNLP (JSON) into the Format-A pickle that
shared/*_utils.py load_*_pkl() reads:  {"train":[...], "dev":[...], "test":[...]}
where each dialogue = {"conv_id":..., "utterance":[...], "speaker":[...], "emotion":[...]}.

Causal ordering is preserved (utterances in dialogue order), so the loader's
_emit_samples builds the next-emotion forecasting samples correctly.

Run on the pod (or anywhere the raw files are):
  python convert/make_pickles.py --meld_dir /path/to/meld_csvs --out meld.pkl --dataset meld
  python convert/make_pickles.py --emory_dir /path/to/emory_jsons --out emorynlp.pkl --dataset emorynlp
"""
import csv, json, pickle, argparse, os
from collections import defaultdict

# ---- canonical label sets (lowercased to match a normalize step) ----
MELD_EMOTIONS  = ["neutral","surprise","fear","sadness","joy","disgust","anger"]
EMORY_EMOTIONS = ["neutral","joyful","powerful","mad","sad","scared","peaceful"]


def build_meld(meld_dir):
    splits = {}
    files = {"train":"train_sent_emo.csv", "dev":"dev_sent_emo.csv", "test":"test_sent_emo.csv"}
    for split, fn in files.items():
        path = os.path.join(meld_dir, fn)
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        # group by Dialogue_ID, order by Utterance_ID
        dias = defaultdict(list)
        for r in rows:
            dias[r["Dialogue_ID"]].append(r)
        dialogues = []
        for did, utts in dias.items():
            utts = sorted(utts, key=lambda r: int(r["Utterance_ID"]))
            dialogues.append({
                "conv_id":  f"meld_{split}_{did}",
                "utterance":[u["Utterance"] for u in utts],
                "speaker":  [u["Speaker"]   for u in utts],
                "emotion":  [u["Emotion"].lower() for u in utts],
            })
        splits[split] = dialogues
        print(f"  MELD {split}: {len(dialogues)} dialogues, {sum(len(d['utterance']) for d in dialogues)} utts")
    return splits


def build_emory(emory_dir):
    splits = {}
    files = {"train":"emotion-detection-trn.json", "dev":"emotion-detection-dev.json", "test":"emotion-detection-tst.json"}
    for split, fn in files.items():
        d = json.load(open(os.path.join(emory_dir, fn), encoding="utf-8"))
        dialogues = []
        for ep in d["episodes"]:
            for sc in ep["scenes"]:
                utts = sc["utterances"]
                dialogues.append({
                    "conv_id":  f"emory_{split}_{sc['scene_id']}",
                    "utterance":[u["transcript"] for u in utts],
                    "speaker":  [u["speakers"][0] if u.get("speakers") else "?" for u in utts],
                    "emotion":  [u["emotion"].lower() for u in utts],
                })
        splits[split] = dialogues
        print(f"  EmoryNLP {split}: {len(dialogues)} scenes, {sum(len(d['utterance']) for d in dialogues)} utts")
    return splits


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["meld","emorynlp"])
    ap.add_argument("--meld_dir", default=".")
    ap.add_argument("--emory_dir", default=".")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.dataset == "meld":
        splits = build_meld(args.meld_dir)
    else:
        splits = build_emory(args.emory_dir)

    with open(args.out, "wb") as f:
        pickle.dump(splits, f)
    print(f"\nSaved -> {args.out}")
