"""
eval_fast.py — fast, no-pipeline evaluation for B4 and B5
=========================================================
Why this exists:
  The run_eval() inside the training scripts uses HuggingFace
  pipeline("text-generation"), which is slow, spams warnings, and looks
  frozen because it prints nothing until 100 samples. This script bypasses
  the pipeline, generates in batches with model.generate(), prints progress
  every batch, and saves the same JSON via shared.evaluate().

  It works for BOTH B4 and B5 because their prompt and parsing are identical.

Usage:
  python eval_fast.py \
      --model_path ./output/b4/final \
      --data_path  /path/to/IEMOCAP.pkl \
      --save_path  ./results/b4.json \
      --model_name "Baseline 4: Direct RL Discrete"

  # quick smoke test on 5 samples:
  python eval_fast.py --model_path ./output/b4/final \
      --data_path /path/to/IEMOCAP.pkl --limit 5 --save_path ./results/b4_smoke.json
"""

import os
import re
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from shared.iemocap_utils import (
    EMOTION_LABELS, LABEL2ID, format_history, load_iemocap_pkl, evaluate,
    make_prompt, parse_emotion, pred_to_id,
)

# Prompt + parsing imported from shared (label-aware, names i_t)



def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--save_path", default=None)
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--model_name", default="Baseline (fast eval)")
    ap.add_argument("--limit", type=int, default=0, help="0 = full split")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    splits = load_iemocap_pkl(args.data_path)
    samples = splits[args.split]
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]
    print(f"Eval split='{args.split}'  samples={len(samples)}", flush=True)

    print(f"Loading model from {args.model_path} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Left padding is required for correct batched decoder-only generation.
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="auto"
    )
    model.eval()

    y_true, y_pred, raw_outputs = [], [], []
    bs = max(1, args.batch_size)
    n = len(samples)

    for start in range(0, n, bs):
        batch = samples[start : start + bs]
        prompts = [make_prompt(s.history, s.history_speakers, s.history_emotions, s.target_speaker) for s in batch]

        enc = tok(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=1024,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )

        gen_only = out[:, enc["input_ids"].shape[1]:]
        decoded = tok.batch_decode(gen_only, skip_special_tokens=True)

        for s, text in zip(batch, decoded):
            emo = parse_emotion(text)
            pid = pred_to_id(emo)
            y_true.append(s.target_emotion_id)
            y_pred.append(pid)
            raw_outputs.append(text.strip())

        done = min(start + bs, n)
        print(f"  [{done}/{n}]", flush=True)

    evaluate(
        y_true, y_pred, args.model_name,
        raw_outputs=raw_outputs, save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
