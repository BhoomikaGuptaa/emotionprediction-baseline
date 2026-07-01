"""
B6 — InstructERC (runnable reimplementation, causal next-emotion)
=================================================================
Faithful to Lei et al., "InstructERC" (arXiv:2309.11911). Reference code:
github.com/LIN-SHANG/InstructERC.

Core pieces reproduced:
  - generative ERC: an instruction-tuned causal LM emits the emotion word,
  - the RETRIEVAL TEMPLATE: instruction + retrieved demonstrations + query, where
    demonstrations are the k most similar TRAIN examples (retrieval over history
    text; TF-IDF here to stay dependency-light and leak-free, train pool only),
  - LoRA fine-tuning.

Adapted to forecasting: the query is the labelled causal history + named next
speaker, and the target is the NEXT emotion (x_t is never shown).

Run:
  python b6_instructerc/run.py --mode train --data_path data/iemocap.pkl \
      --base_model Qwen/Qwen2.5-3B-Instruct --output_dir output/b6
  python b6_instructerc/run.py --mode eval  --data_path data/iemocap.pkl \
      --model_path output/b6/final --save_path results/b6.json
"""
import os, sys, argparse, random, glob
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.iemocap_utils import (
    load_iemocap_pkl, EMOTION_LABELS, LABEL2ID, evaluate, format_history,
    parse_emotion, pred_to_id,
)

INSTRUCTION = (
    "You are an expert in conversational emotion analysis. Given a dialogue history "
    "where each prior turn is tagged with its speaker and emotion, predict the emotion "
    "of the NEXT turn by the named next speaker, without seeing that utterance. "
    f"Answer with exactly one of: {', '.join(EMOTION_LABELS)}."
)


def query_text(s):
    hist = format_history(s.history, s.history_speakers, s.history_emotions, max_turns=10)
    return f"Dialogue history:\n{hist}\nNext speaker: Speaker {s.target_speaker}"


class Retriever:
    """TF-IDF retrieval of k nearest TRAIN demos (train pool only -> no leakage)."""
    def __init__(self, train_samples, k=4):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.k = k
        self.train = train_samples
        self.texts = [query_text(s) for s in train_samples]
        self.vec = TfidfVectorizer(max_features=20000).fit(self.texts)
        self.M = self.vec.transform(self.texts)

    def demos(self, s, exclude_self=True):
        from sklearn.metrics.pairwise import linear_kernel
        q = self.vec.transform([query_text(s)])
        sims = linear_kernel(q, self.M).ravel()
        order = sims.argsort()[::-1]
        # Exclude by conv_id (per CONVERSATION), not dialogue_id (which is per-sample
        # and thus only ever matched the identical sample -- the old leakage bug).
        out, scid = [], s.conv_id
        for j in order:
            if exclude_self and self.train[j].conv_id == scid:
                continue
            out.append(self.train[j])
            if len(out) >= self.k:
                break
        return out


def build_prompt(s, retriever=None):
    demo_block = ""
    if retriever is not None and retriever.k > 0:
        blocks = []
        for d in retriever.demos(s):
            blocks.append(f"{query_text(d)}\nEmotion: {d.target_emotion}")
        demo_block = "Examples:\n" + "\n\n".join(blocks) + "\n\n"
    return f"{INSTRUCTION}\n\n{demo_block}{query_text(s)}\nEmotion:"


def train(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from shared.iemocap_utils import build_config, build_trainer

    if args.load_4bit and not torch.cuda.is_available():
        raise SystemExit(
            "\n[STOP] No GPU detected. QLoRA needs CUDA. In Colab: "
            "Runtime -> Change runtime type -> T4 GPU, then re-run. "
            "If you cannot get a GPU you are likely out of compute units.\n")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    splits = load_iemocap_pkl(args.data_path)
    retr = Retriever(splits["train"], k=args.k)
    rows = [{"text": build_prompt(s, retr) + f" {s.target_emotion}"} for s in splits["train"]]
    ds = Dataset.from_dict({"text": [r["text"] for r in rows]})

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    if args.load_4bit:
        # QLoRA path for small GPUs (e.g. T4 16GB)
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=bnb,
                                                     torch_dtype=torch.float16, device_map="auto")
        model = prepare_model_for_kbit_training(model)
        bf16, fp16 = False, True
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, attn_implementation="eager",
                                                     device_map="auto")
        bf16, fp16 = True, False

    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    model = get_peft_model(model, lora)
    model.config.use_cache = False  # required for gradient checkpointing
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()  # checkpointing + PEFT needs input grads
    if args.load_4bit:
        # Some transformers/peft versions leave LoRA weights in bf16, which the fp16
        # grad scaler cannot unscale ("_amp_foreach...not implemented for BFloat16").
        # Cast trainable params to fp32 (PEFT casts inputs to match, so forward is fine).
        for _p in model.parameters():
            if _p.requires_grad:
                _p.data = _p.data.float()
    model.print_trainable_parameters()

    sft_args = build_config(SFTConfig, output_dir=args.output_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=8, learning_rate=1e-4,
        bf16=bf16, fp16=fp16, logging_steps=20, max_seq_length=args.max_seq_length,
        save_strategy="steps", save_steps=200, save_total_limit=2,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="text", report_to="none")
    trainer = build_trainer(SFTTrainer, model=model, train_dataset=ds, tokenizer=tok, args=sft_args)
    # Resume from the last checkpoint in output_dir if one exists (crash-safe).
    _ckpts = glob.glob(os.path.join(args.output_dir, "checkpoint-*")) if os.path.isdir(args.output_dir) else []
    if _ckpts:
        _last = max(_ckpts, key=lambda p: int(p.split("-")[-1]))
        print(f"[resume] found checkpoint {_last}, resuming training from it")
        trainer.train(resume_from_checkpoint=_last)
    else:
        trainer.train()

    if args.load_4bit:
        # can't cleanly merge a 4-bit base -> save the ADAPTER; eval loads base + adapter
        adapter_dir = os.path.join(args.output_dir, "adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        model.save_pretrained(adapter_dir); tok.save_pretrained(adapter_dir)
        import json as _json
        _json.dump({"base_model": args.base_model, "qlora": True},
                   open(os.path.join(adapter_dir, "adapter_meta.json"), "w"))
        print("saved adapter ->", adapter_dir)
    else:
        adapter_dir = os.path.join(args.output_dir, "adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        model.save_pretrained(adapter_dir); tok.save_pretrained(adapter_dir)
        import json as _json
        _json.dump({"base_model": args.base_model, "qlora": False},
                   open(os.path.join(adapter_dir, "adapter_meta.json"), "w"))
        print("saved adapter ->", adapter_dir)


def run_eval(args):
    import torch, json as _json
    from transformers import AutoTokenizer, AutoModelForCausalLM
    splits = load_iemocap_pkl(args.data_path)
    retr = Retriever(splits["train"], k=args.k)

    if not os.path.isdir(args.model_path):
        raise SystemExit(
            f"\n[STOP] {args.model_path} does not exist, so training did not finish "
            f"(often: no GPU, or the train cell was interrupted). Re-run training on a "
            f"GPU first; eval needs the saved adapter.\n")
    # QLoRA adapter dir (has adapter_meta.json) vs a merged 'final' dir
    meta_path = os.path.join(args.model_path, "adapter_meta.json")
    is_adapter = os.path.exists(meta_path)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    if is_adapter:
        from peft import PeftModel
        meta = _json.load(open(meta_path))
        base = meta["base_model"]
        if meta.get("qlora", False):
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.float16,
                                     bnb_4bit_use_double_quant=True)
            model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb,
                                                         torch_dtype=torch.float16, device_map="auto")
        else:
            # bf16 adapter: load base in bf16 (matches training precision)
            model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                                         device_map="auto")
        model = PeftModel.from_pretrained(model, args.model_path).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, attn_implementation="eager",
                                                     device_map="auto").eval()

    test = splits[args.split]; y_true, y_pred, raw = [], [], []
    bs = args.batch_size
    for i in range(0, len(test), bs):
        batch = test[i:i+bs]
        prompts = [build_prompt(s, retr) for s in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
        dec = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for s, t in zip(batch, dec):
            y_true.append(s.target_emotion_id); y_pred.append(pred_to_id(parse_emotion(t))); raw.append(t.strip())
        print(f"  [{min(i+bs,len(test))}/{len(test)}]", flush=True)
    evaluate(y_true, y_pred, f"B6 InstructERC (causal, seed {args.seed})",
             raw_outputs=raw, save_path=args.save_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["train", "eval"])
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--output_dir", default="output/b6")
    ap.add_argument("--k", type=int, default=4, help="retrieved demonstrations")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_seq_length", type=int, default=1024)
    ap.add_argument("--split", default="test", choices=["train","dev","test"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load_4bit", action="store_true", help="QLoRA 4-bit (for T4/16GB GPUs)")
    ap.add_argument("--save_path", default=None)
    args = ap.parse_args()
    if args.mode == "train": train(args)
    else:
        args.model_path = args.model_path or os.path.join(args.output_dir, "final")
        run_eval(args)


if __name__ == "__main__":
    main()
