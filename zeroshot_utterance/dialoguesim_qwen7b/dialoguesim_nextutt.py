#!/usr/bin/env python3
"""
Dialogue-sim INFERENCE on IEMOCAP test (next-utterance prediction).

Replicates the prompting recipe from kanishkg/dialogue-sim modeling/eval/generate.py
(thinking mode: <think> reasoning then <dialogue> prediction; their generation
settings temperature=0.3, top_p=0.95, max_new_tokens=512), applied to the same
1592 causal IEMOCAP test points as the rest of the project, scored with the same
similarity metrics as the zero-shot baseline.

No training. Inference only. Default model Llama-3.1-8B-Instruct (7-8B class per
the mentor's ask; dialogue-sim's own default was Llama-3.2-3B-Instruct, switchable
via --model).

Usage:
  python dialoguesim_nextutt.py --pkl IEMOCAP_features.pkl \
      --model meta-llama/Llama-3.1-8B-Instruct \
      --out dsim_results.json --preds_out dsim_preds.jsonl
"""
import os, sys, json, argparse, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_nextutt import build_gen_samples, compute_all_metrics

# ---- dialogue-sim's thinking-mode prompt (verbatim from their generate.py) ----
DSIM_SYS = """You are a dialogue prediction system. Your goal is to predict the immediate next dialogue for a conversation in <dialogue> tags. 
Think step by step before you predict the next dialogue.
REMEMBER: The speakers alternate in their dialogues.
Here is how to format your prediction:
<think> Let's think step by step what the next dialogue might be:
[Your reasoning about the context and speakers here]
</think>
<dialogue>
Predicted next dialogue 
</dialogue>"""

def dsim_user(context, next_speaker):
    return (f"Now predict the next dialogue for this conversation. "
            f"Think in <think></think> tags before you predict the dialogue:\n"
            f"{context}\nWhat will the next speaker, {next_speaker}, say?")

def build_context(s, labels_in_history=True):
    """Their format is 'Speaker N: text' lines. We keep our speaker names and,
    matching the project's H_t protocol, optionally show past gold emotion tags."""
    if labels_in_history and s.get("emotions"):
        return "\n".join(f"{sp} ({em}): {ut}" for sp, em, ut
                         in zip(s["speakers"], s["emotions"], s["history"]))
    return "\n".join(f"{sp}: {ut}" for sp, ut in zip(s["speakers"], s["history"]))

DIALOGUE_RE = re.compile(r"<dialogue>\s*(.*?)\s*</dialogue>", re.S | re.I)
def extract_dialogue(text):
    """Their parse: take the <dialogue> tag content.
    Returns (prediction, status):
      'clean'    -> closed <dialogue>...</dialogue> found
      'unclosed' -> <dialogue> opened, never closed (ran out of token budget mid-generation)
      'no_tag'   -> no <dialogue> tag at all (still thinking / ignored format)
    Fallback text is still extracted for unclean cases, but status lets the
    caller report the breakdown honestly instead of scoring fragments silently."""
    m = DIALOGUE_RE.search(text)
    if m:
        out, status = m.group(1), "clean"
    elif "<dialogue>" in text.lower():
        out = re.split(r"<dialogue>", text, flags=re.I)[-1]
        out = re.sub(r"</?\w+>.*", "", out, flags=re.S)
        status = "unclosed"
    else:
        out, status = text.strip().split("\n")[-1], "no_tag"
    out = re.sub(r"^Speaker\s*\w+\s*:\s*", "", out.strip(), flags=re.I)
    return out.strip().strip('"').strip(), status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--labels_in_history", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=512)  # their setting (room to think)
    ap.add_argument("--temperature", type=float, default=0.3)   # their setting
    ap.add_argument("--top_p", type=float, default=0.95)        # their setting
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="dsim_results.json")
    ap.add_argument("--preds_out", default="dsim_preds.jsonl")
    args = ap.parse_args()

    print("=" * 54)
    print("  DIALOGUE-SIM INFERENCE (thinking mode) on IEMOCAP")
    print("=" * 54)

    samples = build_gen_samples(args.pkl, args.split, with_labels=bool(args.labels_in_history))
    if args.limit: samples = samples[:args.limit]
    print(f"{len(samples)} points | model {args.model} | T={args.temperature} top_p={args.top_p}")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"; tok.truncation_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map="auto").eval()
    print(f"loaded in {dtype} | pad={tok.pad_token!r}")

    preds, refs = [], []
    tag_stats = {}
    fp = open(args.preds_out, "w")
    B = args.batch_size
    for i in range(0, len(samples), B):
        batch = samples[i:i+B]
        prompts = []
        for s in batch:
            ctx = build_context(s, bool(args.labels_in_history))
            msgs = [{"role":"system","content":DSIM_SYS},
                    {"role":"user","content":dsim_user(ctx, s["next_speaker"])}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                 max_new_tokens=args.max_new_tokens, do_sample=True,
                                 temperature=args.temperature, top_p=args.top_p,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for s, g in zip(batch, gen):
            p, status = extract_dialogue(g)
            tag_stats[status] = tag_stats.get(status, 0) + 1
            if not p: tag_stats["empty"] = tag_stats.get("empty", 0) + 1
            preds.append(p); refs.append(s["reference"])
            fp.write(json.dumps({"vid":s["vid"],"t":s["t"],"pred":p,"ref":s["reference"],
                                  "status":status, "raw":g[:500]})+"\n")
        if (i // B) % 10 == 0:
            print(f"  {min(i+B,len(samples))}/{len(samples)}", flush=True)
    print(f"  {len(samples)}/{len(samples)} done")
    fp.close()

    n = len(preds)
    print("tag extraction breakdown:")
    for k in ("clean", "unclosed", "no_tag", "empty"):
        v = tag_stats.get(k, 0)
        print(f"  {k:9s}: {v}/{n}  ({100*v/max(n,1):.1f}%)")
    print("  (unclosed/no_tag scored on fallback text; high rate = 512-token thinking budget ran out - reportable finding)")
    print("scoring...")
    scores = compute_all_metrics(preds, refs)
    config = {"method": "dialogue-sim inference (thinking mode)",
              "model": args.model, "split": args.split, "n": len(preds),
              "labels_in_history": bool(args.labels_in_history),
              "temperature": args.temperature, "top_p": args.top_p,
              "max_new_tokens": args.max_new_tokens, "max_hist": 12,
              "seed": args.seed, "tag_stats": tag_stats, "dtype": str(dtype)}
    json.dump({"config": config, "metrics": scores}, open(args.out, "w"), indent=2)
    print(json.dumps(scores, indent=2))

if __name__ == "__main__":
    main()
