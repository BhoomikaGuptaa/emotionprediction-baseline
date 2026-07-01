"""
Baseline 3: Few-shot LLM
========================
Same as B2 but prepends k in-context examples. Exemplars are drawn from the TRAIN
split ONLY (never dev/test) to avoid leakage, sampled deterministically (seeded),
and stratified to cover the label set. Each exemplar shows a labelled history, the
named next speaker, and the gold next emotion in the required output format.

Usage:
  python baseline3_fewshot/run.py --data_path data/iemocap.pkl \
      --backend hf --model meta-llama/Llama-3.1-8B-Instruct \
      --k 6 --save_path results/b3_llama8b.json
"""
import os, sys, argparse, random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.iemocap_utils import (
    load_iemocap_pkl, load_iemocap_json, EMOTION_LABELS, SYSTEM_PROMPT,
    make_prompt, format_history, parse_emotion, pred_to_id, evaluate,
)
from shared.llm_backends import generate_batch


def build_demos(train, k, seed=42):
    """Pick k exemplars from TRAIN, stratified across labels, deterministic."""
    rng = random.Random(seed)
    by_label = {e: [] for e in EMOTION_LABELS}
    for s in train:
        by_label[s.target_emotion].append(s)
    chosen = []
    labels = [e for e in EMOTION_LABELS if by_label[e]]
    i = 0
    while len(chosen) < k and labels:
        lbl = labels[i % len(labels)]
        pool = by_label[lbl]
        if pool:
            chosen.append(pool[rng.randrange(len(pool))])
        i += 1
        if i > 1000:
            break
    blocks = []
    for s in chosen[:k]:
        hist = format_history(s.history, s.history_speakers, s.history_emotions)
        blocks.append(
            f"Dialogue history:\n{hist}\n"
            f"The next turn is spoken by Speaker {s.target_speaker}.\n"
            f"<think>\nbased on the trajectory\n</think>\n"
            f"<emotion>\n{s.target_emotion}\n</emotion>"
        )
    return "\n\n---\n\n".join(blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--loader", default="pkl", choices=["pkl", "json"])
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--backend", default="hf", choices=["hf", "ollama", "openai"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=6, help="number of in-context examples")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--save_path", default=None)
    args = ap.parse_args()

    splits  = load_iemocap_pkl(args.data_path) if args.loader == "pkl" \
        else load_iemocap_json(args.data_path)
    samples = splits[args.split]
    if args.max_samples and args.max_samples > 0:
        samples = samples[:args.max_samples]

    demos = build_demos(splits["train"], args.k, seed=args.seed)
    print(f"Few-shot k={args.k} (from train) | eval {len(samples)} "
          f"| backend={args.backend} model={args.model}")

    y_true, y_pred, raw = [], [], []
    bs = args.batch_size if args.backend == "hf" else 1

    for start in range(0, len(samples), bs):
        batch = samples[start:start + bs]
        prompts = []
        for s in batch:
            q = make_prompt(s.history, s.history_speakers, s.history_emotions,
                            s.target_speaker)
            prompts.append(f"Here are some examples:\n\n{demos}\n\n---\n\nNow your turn.\n\n{q}")
        outs = generate_batch(prompts, args.backend, args.model,
                              system=None, max_new_tokens=args.max_new_tokens,
                              sleep=args.sleep)
        for s, text in zip(batch, outs):
            emo = parse_emotion(text)
            y_true.append(s.target_emotion_id)
            y_pred.append(pred_to_id(emo))
            raw.append(text.strip())
        print(f"  [{min(start+bs, len(samples))}/{len(samples)}]", flush=True)

    evaluate(y_true, y_pred, f"B3 Few-shot (k={args.k}): {args.model}",
             raw_outputs=raw, save_path=args.save_path)


if __name__ == "__main__":
    main()
