#!/usr/bin/env python3
"""
Zero-shot next-utterance generation baseline (IEMOCAP test).
For each point t>=1 in each test dialogue: give the model turns 0..t-1 (with speakers),
ask it to produce the next utterance for the known next speaker, score the generation
against the REAL utterance u_t with a full suite of similarity metrics.

No training. Inference only.
"""
import os, sys, json, argparse, pickle, re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------- build causal generation samples straight from the pkl ----------
def load_iemocap_raw(pkl_path):
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    return raw

NUM2EMO = {0:"happiness",1:"sadness",2:"neutral",3:"anger",4:"excited",5:"frustration"}

def build_gen_samples(pkl_path, split="test", max_hist=12, with_labels=True):
    """Return list of dicts: {history, speakers, emotions(optional), next_speaker, reference}."""
    raw = load_iemocap_raw(pkl_path)
    # Standard DialogueRNN IEMOCAP pkl layout:
    # raw[2]=speakers, raw[3? ] varies; use the videoSentence + videoSpeakers + split ids form.
    # Most common: raw = [videoIDs, videoSpeakers, videoLabels, videoText, videoAudio,
    #                      videoVisual, videoSentence, trainVid, testVid]
    if isinstance(raw, (list, tuple)) and len(raw) >= 9:
        videoSpeakers = raw[1]
        videoLabels   = raw[2]
        videoSentence = raw[6]
        trainVid, testVid = raw[7], raw[8]
    elif isinstance(raw, dict):
        videoSpeakers = raw["videoSpeakers"]; videoSentence = raw["videoSentence"]
        videoLabels = raw.get("videoLabels", {})
        trainVid = raw.get("trainVid"); testVid = raw.get("testVid")
    else:
        raise ValueError("Unrecognized IEMOCAP pkl format")

    ids = list(testVid) if split == "test" else list(trainVid)
    samples = []
    for vid in ids:
        sents = videoSentence[vid]
        spks_raw = videoSpeakers[vid]
        # speakers may be one-hot [1,0]/[0,1] or 'M'/'F' -> normalize to a label
        def spk_label(s):
            if isinstance(s, (list, tuple, np.ndarray)):
                return "SpeakerA" if int(np.argmax(s)) == 0 else "SpeakerB"
            return str(s)
        spks = [spk_label(s) for s in spks_raw]
        labs = videoLabels.get(vid, []) if isinstance(videoLabels, dict) else []
        for t in range(1, len(sents)):
            hist = sents[:t]
            hspk = spks[:t]
            hemo = [NUM2EMO.get(int(l), str(l)) for l in labs[:t]] if (with_labels and len(labs) >= t) else None
            samples.append({
                "vid": vid, "t": t,
                "history": hist[-max_hist:],
                "speakers": hspk[-max_hist:],
                "emotions": (hemo[-max_hist:] if hemo else None),
                "next_speaker": spks[t],
                "reference": sents[t],
            })
    return samples

# ---------- prompt ----------
SYS = ("You continue conversations. Given the dialogue so far, write ONLY the next line "
       "that the named speaker would say. Output just the utterance text, nothing else, "
       "no speaker name, no quotes, no explanation.")

def build_prompt(s):
    if s.get("emotions"):
        lines = [f"{sp} ({em}): {ut}" for sp, em, ut in zip(s["speakers"], s["emotions"], s["history"])]
    else:
        lines = [f"{sp}: {ut}" for sp, ut in zip(s["speakers"], s["history"])]
    convo = "\n".join(lines)
    return (f"{convo}\n{s['next_speaker']}:")

def clean_gen(text, next_speaker):
    """Strip common preambles/speaker prefixes/quotes so scoring is fair."""
    t = text.strip()
    # remove leading "SpeakerX:" if the model echoed it
    t = re.sub(rf"^{re.escape(next_speaker)}\s*:\s*", "", t, flags=re.I)
    t = re.sub(r"^(sure|here'?s?|the next line( is)?|response)\s*[:,-]?\s*", "", t, flags=re.I)
    t = t.strip().strip('"').strip()
    # keep only first line (utterance, not a monologue)
    t = t.split("\n")[0].strip()
    return t

# ---------- main ----------
def main():
    print("=" * 50)
    print("  SCRIPT VERSION: v3 (pad/attention-mask fix)")
    print("=" * 50)
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="0 = all; else first N for a quick test")
    ap.add_argument("--load_8bit", action="store_true", help="8-bit load for 16GB GPUs (T4). Needs bitsandbytes.")
    ap.add_argument("--batch_size", type=int, default=8, help="generation batch size; use 2-4 on a 16GB T4")
    ap.add_argument("--parrot_floor", action="store_true",
                    help="No model: 'predict' the next utterance by repeating the LAST utterance in history. "
                         "The generation analog of the persistence floor. Zero GPU.")
    ap.add_argument("--labels_in_history", type=int, default=1,
                    help="1 = show past turns' gold emotion tags in the prompt (matches the project's H_t protocol). "
                         "0 = text+speaker only. The trained dialogue-sim baseline MUST use the same setting.")
    ap.add_argument("--out", default="nextutt_zeroshot_results.json")
    ap.add_argument("--preds_out", default="nextutt_zeroshot_preds.jsonl")
    args = ap.parse_args()

    samples = build_gen_samples(args.pkl, args.split, with_labels=bool(args.labels_in_history))
    if args.parrot_floor:
        # No model load at all: prediction = last utterance in history.
        preds = [s_["history"][-1] for s_ in samples] if not args.limit else [s_["history"][-1] for s_ in samples[:args.limit]]
        refs  = [s_["reference"] for s_ in samples] if not args.limit else [s_["reference"] for s_ in samples[:args.limit]]
        with open(args.preds_out, "w") as fp:
            for s_, p in zip(samples[:len(preds)], preds):
                fp.write(json.dumps({"vid":s_["vid"],"t":s_["t"],"pred":p,"ref":s_["reference"]})+"\n")
        print(f"[parrot floor] {len(preds)} points, scoring...")
        scores = compute_all_metrics(preds, refs)
        config = {"model": "PARROT_FLOOR (repeat last utterance)", "split": args.split, "n": len(preds),
                  "labels_in_history": bool(args.labels_in_history), "max_hist": 12}
        json.dump({"config": config, "metrics": scores}, open(args.out,"w"), indent=2)
        print(json.dumps(scores, indent=2)); return
    # Comparability check: this script builds points directly from the raw pkl. Verify the
    # (vid, t) point set matches the shared emotion-task loader exactly, so utterance metrics
    # and emotion metrics are over the IDENTICAL points.
    try:
        os.environ.setdefault("ERC_DATASET", "iemocap")
        from shared.iemocap_utils import load_iemocap_pkl
        sh = load_iemocap_pkl(args.pkl)[args.split]
        shared_pts = {(str(x.conv_id), len(x.history)) for x in sh} if hasattr(sh[0], "conv_id") else None
        mine_pts   = {(str(s_["vid"]), s_["t"]) for s_ in samples}
        if shared_pts is not None:
            if shared_pts == mine_pts:
                print(f"[check] point set matches shared loader exactly ({len(mine_pts)} points)")
            else:
                only_mine = len(mine_pts - shared_pts); only_shared = len(shared_pts - mine_pts)
                print(f"[WARN] point sets differ: {only_mine} only here, {only_shared} only in shared loader.")
                print("[WARN] utterance vs emotion metrics will NOT be over identical points. Investigate before trusting comparisons.")
    except Exception as e:
        print(f"[check skipped] could not compare with shared loader: {e}")
    if args.limit: samples = samples[:args.limit]
    print(f"{len(samples)} generation points from {args.split}")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"loading {args.model} in {dtype}" + (" (8-bit)" if args.load_8bit else ""))
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"      # decoder-only models REQUIRE left padding for batched generation
    tok.truncation_side = "left"   # if a prompt is too long, drop the OLDEST turns, never the recent ones or the speaker cue
    if args.load_8bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, device_map="auto").eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map="auto").eval()
    if tok.pad_token is None:
        # Qwen2.5 has <|endoftext|> as a distinct pad-capable token; prefer it over eos
        try:
            tok.pad_token = "<|endoftext|>"
            _ = tok.pad_token_id  # force resolve
        except Exception:
            tok.pad_token = tok.eos_token
    print(f"pad_token={tok.pad_token!r} (id {tok.pad_token_id}), eos={tok.eos_token!r} (id {tok.eos_token_id})")

    preds, refs = [], []
    fpred = open(args.preds_out, "w")
    B = args.batch_size
    for i in range(0, len(samples), B):
        batch = samples[i:i+B]
        prompts = []
        for s in batch:
            msgs = [{"role":"system","content":SYS},
                    {"role":"user","content":build_prompt(s)}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1536).to(model.device)
        with torch.no_grad():
            out = model.generate(input_ids=enc["input_ids"],
                                 attention_mask=enc["attention_mask"],
                                 max_new_tokens=args.max_new_tokens,
                                 do_sample=False,
                                 repetition_penalty=1.1,
                                 pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id)
        gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for s, g in zip(batch, gen):
            p = clean_gen(g, s["next_speaker"])
            preds.append(p); refs.append(s["reference"])
            fpred.write(json.dumps({"vid":s["vid"],"t":s["t"],"pred":p,"ref":s["reference"]})+"\n")
        if (i//B) % 10 == 0:
            print(f"  {i+len(batch)}/{len(samples)}", flush=True)
    fpred.close()

    print("scoring...")
    scores = compute_all_metrics(preds, refs)
    config = {"model": args.model, "split": args.split, "n": len(preds),
              "labels_in_history": bool(args.labels_in_history), "load_8bit": bool(args.load_8bit),
              "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
              "max_hist": 12, "dtype": str(dtype)}
    json.dump({"config": config, "metrics": scores}, open(args.out,"w"), indent=2)
    print(json.dumps(scores, indent=2))

def compute_all_metrics(preds, refs):
    out = {}
    # sacrebleu: BLEU + chrF
    try:
        import sacrebleu
        out["bleu"]  = sacrebleu.corpus_bleu(preds, [refs]).score
        out["chrf"]  = sacrebleu.corpus_chrf(preds, [refs]).score
    except Exception as e:
        out["bleu_error"]=str(e)
    # rouge
    try:
        from rouge_score import rouge_scorer
        sc = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
        r1=r2=rL=0.0
        for p,r in zip(preds,refs):
            s=sc.score(r,p); r1+=s["rouge1"].fmeasure; r2+=s["rouge2"].fmeasure; rL+=s["rougeL"].fmeasure
        n=len(preds); out["rouge1"]=r1/n; out["rouge2"]=r2/n; out["rougeL"]=rL/n
    except Exception as e:
        out["rouge_error"]=str(e)
    # meteor
    try:
        import nltk; nltk.download("wordnet", quiet=True); nltk.download("punkt", quiet=True)
        from nltk.translate.meteor_score import meteor_score
        m=0.0
        for p,r in zip(preds,refs):
            m+=meteor_score([r.split()], p.split())
        out["meteor"]=m/len(preds)
    except Exception as e:
        out["meteor_error"]=str(e)
    # BERTScore
    try:
        from bert_score import score as bertscore
        P,R,F = bertscore(preds, refs, lang="en", verbose=False, rescale_with_baseline=True)
        out["bertscore_f1"]=float(F.mean())
    except Exception as e:
        out["bertscore_error"]=str(e)
    # SBERT cosine
    try:
        from sentence_transformers import SentenceTransformer, util
        st = SentenceTransformer("all-mpnet-base-v2")
        ep = st.encode(preds, convert_to_tensor=True, show_progress_bar=False)
        er = st.encode(refs,  convert_to_tensor=True, show_progress_bar=False)
        cos = util.cos_sim(ep, er).diagonal()
        out["sbert_cosine"]=float(cos.mean())
    except Exception as e:
        out["sbert_error"]=str(e)
    return out

if __name__ == "__main__":
    main()
