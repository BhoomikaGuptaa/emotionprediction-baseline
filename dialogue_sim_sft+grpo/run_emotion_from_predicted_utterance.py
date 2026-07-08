"""
Emotion recognition on PREDICTED next-utterances (generate-then-classify pipeline)
====================================================================================
Given the causal history H_t (same as every other baseline: turns 0..t-1, each
tagged with speaker and gold emotion), plus a PREDICTED utterance for turn t
(from an utterance-generation checkpoint - here, SFT-only), classify the emotion
that predicted utterance expresses. Score against the GOLD emotion of the real
turn t (never the gold utterance text - only the predicted one is shown).

This is functionally the emotion-FORECASTING pipeline requested: you never see
the ground-truth utterance, only a predicted stand-in for it, so accuracy here
reflects the full generate-then-classify chain, not just classification quality.

Reuses B2's exact SYSTEM_PROMPT reply format (<think>/<emotion> tags), parse_emotion,
pred_to_id, and evaluate() UNMODIFIED, so this row is scored identically to every
other emotion baseline in the project - only the prompt CONTENT changes (recognition
of a given utterance vs. blind prediction from history alone), not the scoring code.

Usage:
  python run_emotion_from_predicted_utterance.py \
      --data_path /path/to/IEMOCAP_features.pkl \
      --preds_path /path/to/sft_only_preds_stripped.jsonl \
      --backend hf --model meta-llama/Llama-3.2-1B-Instruct \
      --save_path results/emotion_from_sft_predicted_utterance.json
"""
import os, sys, json, re, argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from shared.iemocap_utils import (
    load_iemocap_pkl, load_iemocap_json, format_history, EMOTION_LABELS,
    parse_emotion, pred_to_id, evaluate,
)
from shared.llm_backends import generate_batch


# ---- Recognition-mode prompt: SAME reply format as B2, different task framing ----
# B2's SYSTEM_PROMPT asks the model to predict blind (never sees turn t).
# This one shows a PREDICTED utterance for turn t and asks what emotion IT expresses -
# a recognition task, not a blind-prediction task. Keeping the exact same
# <think>/<emotion> reply format means parse_emotion/pred_to_id/evaluate need
# zero changes.
SYSTEM_PROMPT_RECOG = (
    "You are an expert in conversational emotion analysis.\n"
    "You are given a dialogue history. Each prior turn is labelled with the "
    "speaker and the emotion they expressed. You are then given a NEW utterance "
    "spoken by the next speaker.\n"
    "Predict the emotion expressed by that new utterance.\n\n"
    f"Valid emotions: {', '.join(EMOTION_LABELS)}\n\n"
    "Reply in this EXACT format:\n"
    "<think>\nbrief reasoning about the emotional content of the utterance\n</think>\n"
    "<emotion>\none_emotion_label\n</emotion>"
)


def make_prompt_recognition(history, speakers, emotions, target_speaker,
                            target_utterance, max_turns=10):
    hist = format_history(history, speakers, emotions, max_turns=max_turns)
    return (
        f"{SYSTEM_PROMPT_RECOG}\n\n"
        f"Dialogue history:\n{hist}\n\n"
        f"Speaker {target_speaker} now says: \"{target_utterance}\"\n"
        f"What emotion does this utterance express?"
    )


DID_RE = re.compile(r"^(.*)_t(\d+)$")


def parse_dialogue_id(dialogue_id):
    """Sample.dialogue_id is f'{vid}_t{t}'. vid itself may contain underscores
    (e.g. 'Ses05M_impro06'), so match on the LAST '_t<digits>' suffix, not a
    naive split on the first underscore."""
    m = DID_RE.match(dialogue_id)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def load_predicted_utterances(path):
    """Load a *_preds_stripped.jsonl (vid, t, pred, ref, tag_status fields) into
    a {(vid, t): pred_text} lookup."""
    lookup = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (str(r.get("vid")), r.get("t"))
            lookup[key] = r.get("pred", "") or ""
    return lookup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--loader", default="pkl", choices=["pkl", "json"])
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--preds_path", required=True,
                    help="the *_preds_stripped.jsonl from score_rl_generations.py "
                         "(vid/t/pred fields) - e.g. sft_only_preds_stripped.jsonl")
    ap.add_argument("--backend", default="hf", choices=["hf", "ollama", "openai"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = full split")
    ap.add_argument("--batch_size", type=int, default=16, help="hf batch size")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--sleep", type=float, default=0.0, help="delay between API calls")
    ap.add_argument("--model_name_label", default=None,
                    help="label for the results json/printout; default derived from --model")
    ap.add_argument("--save_path", default=None)
    args = ap.parse_args()

    splits = load_iemocap_pkl(args.data_path) if args.loader == "pkl" \
        else load_iemocap_json(args.data_path)
    samples = splits[args.split]
    if args.max_samples and args.max_samples > 0:
        samples = samples[:args.max_samples]

    pred_lookup = load_predicted_utterances(args.preds_path)
    print(f"[load] {len(samples)} samples from {args.split}, "
          f"{len(pred_lookup)} predicted utterances from {args.preds_path}")

    # join check - same discipline as every other point-set comparison in this project
    joined, missing = 0, 0
    for s in samples:
        vid, t = parse_dialogue_id(s.dialogue_id)
        if (vid, t) in pred_lookup:
            joined += 1
        else:
            missing += 1
    print(f"[join] {joined} matched, {missing} missing "
          f"({100*missing/max(1,len(samples)):.1f}% of points)")
    if missing:
        print("[warn] missing points will still be scored (empty utterance -> "
              "model gets no content to classify, scored as a genuine attempt, "
              "not dropped) - point count stays comparable to every other row.")

    print(f"backend={args.backend} model={args.model}")

    y_true, y_pred, raw = [], [], []
    bs = args.batch_size if args.backend == "hf" else 1

    for start in range(0, len(samples), bs):
        batch = samples[start:start + bs]
        prompts = []
        for s in batch:
            vid, t = parse_dialogue_id(s.dialogue_id)
            pred_utt = pred_lookup.get((vid, t), "")
            prompts.append(make_prompt_recognition(
                s.history, s.history_speakers, s.history_emotions,
                s.target_speaker, pred_utt,
            ))
        outs = generate_batch(prompts, args.backend, args.model,
                              system=None, max_new_tokens=args.max_new_tokens,
                              sleep=args.sleep)
        for s, text in zip(batch, outs):
            emo = parse_emotion(text)
            y_true.append(s.target_emotion_id)
            y_pred.append(pred_to_id(emo))
            raw.append(text.strip())
        print(f"  [{min(start+bs, len(samples))}/{len(samples)}]", flush=True)

    model_label = args.model_name_label or (
        f"Emotion-from-predicted-utterance ({args.model}, "
        f"utterances from {os.path.basename(args.preds_path)})"
    )
    evaluate(y_true, y_pred, model_label, raw_outputs=raw, save_path=args.save_path)


if __name__ == "__main__":
    main()
