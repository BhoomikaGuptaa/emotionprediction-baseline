"""
Baseline 2: Zero-shot LLM
=========================
Prompts an instruction LLM to predict the next emotion from the labelled history,
with NO in-context examples. Greedy decoding.

To line up with the DPM paper's zero-shot rows, run with:
  --backend hf --model meta-llama/Llama-3.2-1B-Instruct
  --backend hf --model meta-llama/Llama-3.1-8B-Instruct
You may also keep gpt-4o-mini / qwen rows as extras.

Usage:
  python baseline2_zeroshot/run.py --data_path data/iemocap.pkl \
      --backend hf --model meta-llama/Llama-3.2-1B-Instruct \
      --save_path results/b2_llama1b.json
"""
import os, sys, argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.iemocap_utils import (
    load_iemocap_pkl, load_iemocap_json, SYSTEM_PROMPT, make_prompt,
    parse_emotion, pred_to_id, evaluate,
)
from shared.llm_backends import generate_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--loader", default="pkl", choices=["pkl", "json"])
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--backend", default="hf", choices=["hf", "ollama", "openai"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = full split")
    ap.add_argument("--batch_size", type=int, default=16, help="hf batch size")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--sleep", type=float, default=0.0, help="delay between API calls")
    ap.add_argument("--save_path", default=None)
    args = ap.parse_args()

    splits  = load_iemocap_pkl(args.data_path) if args.loader == "pkl" \
        else load_iemocap_json(args.data_path)
    samples = splits[args.split]
    if args.max_samples and args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f"Eval split={args.split} samples={len(samples)} "
          f"backend={args.backend} model={args.model}")

    y_true, y_pred, raw = [], [], []
    bs = args.batch_size if args.backend == "hf" else 1

    for start in range(0, len(samples), bs):
        batch   = samples[start:start + bs]
        prompts = [make_prompt(s.history, s.history_speakers, s.history_emotions,
                               s.target_speaker) for s in batch]
        outs = generate_batch(prompts, args.backend, args.model,
                              system=None, max_new_tokens=args.max_new_tokens,
                              sleep=args.sleep)
        for s, text in zip(batch, outs):
            emo = parse_emotion(text)
            y_true.append(s.target_emotion_id)
            y_pred.append(pred_to_id(emo))
            raw.append(text.strip())
        print(f"  [{min(start+bs, len(samples))}/{len(samples)}]", flush=True)

    evaluate(y_true, y_pred, f"B2 Zero-shot: {args.model}",
             raw_outputs=raw, save_path=args.save_path)


if __name__ == "__main__":
    main()
