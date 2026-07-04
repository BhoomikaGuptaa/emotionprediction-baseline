#!/usr/bin/env python3
"""
Next-utterance prediction via the OpenAI API (same task, same prompts, same metrics
as generate_nextutt.py, but the generator is an OpenAI chat model instead of a local model).

Usage:
  export OPENAI_API_KEY=sk-...
  python openai_nextutt.py --pkl IEMOCAP_features.pkl --model gpt-4o-mini \
      --out nextutt_openai_results.json --preds_out nextutt_openai_preds.jsonl

Cost (approx, full 1592-point IEMOCAP test):
  gpt-4o-mini: ~$0.15-0.25 total   |   gpt-4o: ~$2-3 total
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_nextutt import build_gen_samples, build_prompt, clean_gen, compute_all_metrics, SYS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--labels_in_history", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--out", default="nextutt_openai_results.json")
    ap.add_argument("--preds_out", default="nextutt_openai_preds.jsonl")
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY from env

    samples = build_gen_samples(args.pkl, args.split, with_labels=bool(args.labels_in_history))
    if args.limit: samples = samples[:args.limit]
    print(f"{len(samples)} generation points from {args.split} | model {args.model}")

    preds, refs = [], []
    fp = open(args.preds_out, "w")
    for i, s in enumerate(samples):
        for attempt in range(4):
            try:
                r = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "system", "content": SYS},
                              {"role": "user", "content": build_prompt(s)}],
                    max_tokens=args.max_new_tokens,
                    temperature=0,
                )
                g = r.choices[0].message.content or ""
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"  retry in {wait}s: {e}")
                time.sleep(wait)
        else:
            g = ""
        p = clean_gen(g, s["next_speaker"])
        preds.append(p); refs.append(s["reference"])
        fp.write(json.dumps({"vid": s["vid"], "t": s["t"], "pred": p, "ref": s["reference"]}) + "\n")
        if (i + 1) % 50 == 0: print(f"  {i+1}/{len(samples)}", flush=True)
    fp.close()

    print("scoring...")
    scores = compute_all_metrics(preds, refs)
    config = {"model": args.model, "api": "openai", "split": args.split, "n": len(preds),
              "labels_in_history": bool(args.labels_in_history),
              "max_new_tokens": args.max_new_tokens, "max_hist": 12, "temperature": 0}
    json.dump({"config": config, "metrics": scores}, open(args.out, "w"), indent=2)
    print(json.dumps(scores, indent=2))

if __name__ == "__main__":
    main()
