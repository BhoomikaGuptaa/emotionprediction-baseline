"""
Baseline 4: Direct RL — Discrete Reward
========================================
Task:    history u1...u(t-1)  →  emotion of u(t)
Model:   Qwen2.5-3B-Instruct + LoRA, trained with SFT → GRPO
Reward:  three-component rule-based, max = 1.0

    R_format      = 0.2   <think>...</think><emotion>...</emotion> present
    R_valid_label = 0.2   output is one of the 6 valid emotion words
    R_correct     = 0.6   predicted emotion == ground truth
    ─────────────────────────────────────────────────────────
    Total max     = 1.0

Based on:
  - kanishkg/dialogue-sim  (SFT → GRPO two-stage pipeline, <think> format)
  - DeepSeek-R1 / RLVR    (rule-based verifiable reward pattern)
  - EMO-RL (Li et al. 2025) (format + accuracy reward decomposition)

Usage:
  # Step 1: inspect your pkl first
  python b4_discrete.py --mode inspect --data_path /path/to/IEMOCAP.pkl

  # Step 2: train
  python b4_discrete.py --mode train --data_path /path/to/IEMOCAP.pkl

  # Step 3: evaluate
  python b4_discrete.py --mode eval --data_path /path/to/IEMOCAP.pkl \
      --model_path ./output/b4_final --save_path ./results/b4.json
"""
# --- compat shim (must run before any `trl` import) -------------------------
# The NGC container ships torch 2.5, whose torch.distributed.fsdp has no
# FSDPModule. Newer TRL imports it for FSDP2 support. We run single-GPU with
# no FSDP, so inject a harmless dummy so the GRPO import chain succeeds.
try:
    import torch.distributed.fsdp as _fsdp
    if not hasattr(_fsdp, "FSDPModule"):
        class _FSDPModuleShim:  # never instantiated on single-GPU non-FSDP runs
            pass
        _fsdp.FSDPModule = _FSDPModuleShim
except Exception:
    pass
try:
    import torch.distributed.tensor as _dtensor
    if not hasattr(_dtensor, "DTensor"):
        class _DTensorShim:  # single-GPU weights are never DTensors, so isinstance -> False
            pass
        _dtensor.DTensor = _DTensorShim
except Exception:
    pass
# ---------------------------------------------------------------------------


import re
import os
import sys
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.iemocap_utils import (
    EMOTION_LABELS, LABEL2ID, ID2LABEL, NUM_LABELS,
    format_history, load_iemocap_pkl, inspect_pkl, evaluate,
    build_config, build_trainer,
    make_prompt, parse_emotion, check_format, pred_to_id,
)

# ── Model config ──────────────────────────────────────────────────────────────
BASE_MODEL    = "Qwen/Qwen2.5-3B-Instruct"   # change to 1.5B if low VRAM
OUTPUT_DIR    = "./output/b4"
SFT_DIR       = "./output/b4/sft_checkpoint"

# ── Training hyperparameters (from EMO-RL paper + kanishkg) ──────────────────
SFT_LR         = 1e-5
SFT_EPOCHS     = 2
GRPO_LR        = 1e-6
GRPO_STEPS     = 300
GRPO_G         = 2        # candidates per prompt (set 6 later if approved)
GRPO_BETA      = 0.001    # KL coefficient
MAX_SEQ_LEN    = 1024     # SFT
MAX_NEW_TOKENS = 128      # GRPO generation
BATCH_SIZE     = 1
GRAD_ACCUM     = 4
LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05


# Prompt (make_prompt) and parsing (parse_emotion/check_format/pred_to_id) are
# imported from shared.iemocap_utils. They are label-aware: each history turn shows
# its gold emotion and the prompt names the next speaker i_t.


def completion_to_text(completion):
    """Normalize TRL completion objects across TRL versions."""
    if isinstance(completion, list):
        return " ".join(m.get("content", "") for m in completion if isinstance(m, dict))
    if isinstance(completion, dict):
        return completion.get("content", str(completion))
    return str(completion)


def expand_targets(targets, n):
    """Expand one target per prompt to one target per generated completion."""
    if targets is None:
        return ["neutral"] * n
    if not isinstance(targets, (list, tuple)):
        return [targets] * n
    targets = list(targets)
    if len(targets) == n:
        return targets
    if len(targets) > 0 and n % len(targets) == 0:
        reps = n // len(targets)
        return [t for t in targets for _ in range(reps)]
    if len(targets) > 0:
        return [targets[i % len(targets)] for i in range(n)]
    return ["neutral"] * n


# ── Reward function ───────────────────────────────────────────────────────────
def reward_b4(generated: str, target: str) -> float:
    """
    B4 reward — discrete binary, three components, max = 1.0.

    R_format      = 0.2
    R_valid_label = 0.2
    R_correct     = 0.6
    """
    r   = 0.0
    pred = parse_emotion(generated)
    if check_format(generated):   r += 0.2
    if pred is not None:          r += 0.2
    if pred == target:            r += 0.6
    return r


# ── Dataset helpers ───────────────────────────────────────────────────────────
def make_sft_rows(samples):
    rows = []
    for s in samples:
        prompt = make_prompt(s.history, s.history_speakers, s.history_emotions, s.target_speaker)
        completion = (
            "<think>\n"
            "Analysing the emotional trajectory of this conversation.\n"
            "</think>\n"
            f"<emotion>\n{s.target_emotion}\n</emotion>"
        )
        rows.append({"prompt": prompt, "completion": completion,
                     "text": prompt + "\n" + completion})
    return rows


def make_grpo_rows(samples):
    return [
        {"prompt": make_prompt(s.history, s.history_speakers, s.history_emotions, s.target_speaker),
         "target_emotion": s.target_emotion}
        for s in samples
    ]


# ── Training ──────────────────────────────────────────────────────────────────
def _last_ckpt(d):
    """Return latest checkpoint dir if one exists, else None (so first run starts fresh)."""
    import os
    from transformers.trainer_utils import get_last_checkpoint
    return get_last_checkpoint(d) if os.path.isdir(d) else None


def train(data_path, base_model=BASE_MODEL, output_dir=OUTPUT_DIR, loader="pkl",
          max_train=0, grpo_steps=0, grpo_g=0, seed=42):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig
        from datasets import Dataset
    except ImportError:
        raise ImportError(
            "pip install transformers peft trl datasets accelerate bitsandbytes"
        )

    grpo_steps = grpo_steps or GRPO_STEPS
    grpo_g = grpo_g or GRPO_G
    import random, numpy as _np
    random.seed(seed)
    _np.random.seed(seed)
    import torch as _t; _t.manual_seed(seed)

    splits = load_iemocap_pkl(data_path) if loader == "pkl" \
             else __import__("shared.iemocap_utils", fromlist=["load_iemocap_json"]).load_iemocap_json(data_path)

    train_s = splits["train"]
    dev_s   = splits["dev"]
    test_s  = splits["test"]
    if max_train and max_train > 0:
        train_s = train_s[:max_train]
        print(f"[debug] capping train set to {len(train_s)} samples")
    print(f"\nTrain: {len(train_s)} | Dev: {len(dev_s)} | Test: {len(test_s)}")

    # Label distribution check
    from collections import Counter
    dist = Counter(s.target_emotion for s in train_s)
    print("\nTraining label distribution:")
    for lbl, cnt in dist.most_common():
        print(f"  {lbl:<15} {cnt:4d}  ({100*cnt/len(train_s):.1f}%)")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    sft_dir = os.path.join(output_dir, "sft_checkpoint")
    os.makedirs(sft_dir, exist_ok=True)

    # ── Stage 1: SFT warmup ───────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Stage 1: SFT Warmup")
    print(f"{'='*55}")
    print("Purpose: teach the model the <think>/<emotion> output format")
    print(f"Epochs: {SFT_EPOCHS} | LR: {SFT_LR}")

    sft_data    = make_sft_rows(train_s)
    sft_dataset = Dataset.from_dict({"text": [d["text"] for d in sft_data]})

    sft_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16, attn_implementation="eager",
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        task_type    = TaskType.CAUSAL_LM,
        r            = LORA_R,
        lora_alpha   = LORA_ALPHA,
        lora_dropout = LORA_DROPOUT,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
    )
    sft_model = get_peft_model(sft_model, lora_cfg)
    sft_model.print_trainable_parameters()

    sft_args = build_config(
        SFTConfig,
        output_dir                  = sft_dir,
        num_train_epochs            = SFT_EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = SFT_LR,
        bf16                        = True,
        logging_steps               = 20,
        save_strategy               = "epoch",
        save_total_limit            = 1,
        max_seq_length              = MAX_SEQ_LEN,   # aliased to max_length on new TRL
        dataset_text_field          = "text",
        report_to                   = "none",
    )
    sft_trainer = build_trainer(
        SFTTrainer,
        model         = sft_model,
        train_dataset = sft_dataset,
        tokenizer     = tokenizer,                   # aliased to processing_class on new TRL
        args          = sft_args,
    )
    sft_trainer.train(resume_from_checkpoint=_last_ckpt(sft_dir))

    # Merge LoRA before saving so GRPO can load cleanly
    sft_model = sft_model.merge_and_unload()
    sft_model.save_pretrained(sft_dir)
    tokenizer.save_pretrained(sft_dir)
    print(f"\nSFT saved → {sft_dir}")
    del sft_model, sft_trainer
    torch.cuda.empty_cache()

    # ── Stage 2: GRPO ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Stage 2: GRPO Reinforcement Learning")
    print(f"{'='*55}")
    print(f"Reward: R_format(0.2) + R_valid_label(0.2) + R_correct(0.6)")
    print(f"Steps: {grpo_steps} | LR: {GRPO_LR} | G={GRPO_G} | β={GRPO_BETA}")

    grpo_model = AutoModelForCausalLM.from_pretrained(
        sft_dir,
        torch_dtype=torch.bfloat16, attn_implementation="eager",
        device_map  = "auto",
    )

    # Apply LoRA again for GRPO fine-tuning
    grpo_model = get_peft_model(grpo_model, lora_cfg)

    grpo_rows    = make_grpo_rows(train_s)
    grpo_dataset = Dataset.from_dict({
        "prompt":         [d["prompt"]         for d in grpo_rows],
        "target_emotion": [d["target_emotion"] for d in grpo_rows],
    })

    def reward_fn(completions, prompts=None, **kwargs):
        """
        TRL GRPOTrainer reward function interface.
        completions: list of generated strings (one per sample * G)
        kwargs contains extra columns from the dataset, including target_emotion
        """
        texts = [completion_to_text(c) for c in completions]
        targets = expand_targets(kwargs.get("target_emotion"), len(texts))
        return [reward_b4(t, g) for t, g in zip(texts, targets)]

    grpo_args = build_config(
        GRPOConfig,
        output_dir                  = output_dir,
        max_steps                   = grpo_steps,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = GRPO_LR,
        num_generations             = grpo_g,
        max_completion_length       = MAX_NEW_TOKENS,
        temperature                 = 1.0,
        beta                        = GRPO_BETA,
        bf16                        = True,
        logging_steps               = 10,
        save_steps                  = 100,
        save_total_limit            = 1,
        report_to                   = "none",
    )
    grpo_trainer = build_trainer(
        GRPOTrainer,
        model         = grpo_model,
        tokenizer     = tokenizer,
        train_dataset = grpo_dataset,
        reward_funcs  = reward_fn,
        args          = grpo_args,
    )
    grpo_trainer.train(resume_from_checkpoint=_last_ckpt(output_dir))

    # Merge and save final model
    final_model = grpo_model.merge_and_unload()
    final_dir   = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    final_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nFinal model saved → {final_dir}")

    # Save reward config for reproducibility
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump({
            "baseline": "B4",
            "reward": {"format": 0.2, "valid_label": 0.2, "correct": 0.6},
            "grpo_steps": GRPO_STEPS, "grpo_lr": GRPO_LR,
            "grpo_G": grpo_g, "beta": GRPO_BETA, "seed": seed,
            "base_model": base_model,
        }, f, indent=2)


# ── Evaluation ────────────────────────────────────────────────────────────────
def run_eval(model_path, data_path, loader="pkl", split="test", save_path=None):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    except ImportError:
        raise ImportError("pip install transformers torch")

    splits  = load_iemocap_pkl(data_path) if loader == "pkl" else \
              __import__("shared.iemocap_utils", fromlist=["load_iemocap_json"]).load_iemocap_json(data_path)
    samples = splits[split]

    print(f"\nLoading model from {model_path}...")
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="auto"
    )
    gen = pipeline(
        "text-generation", model=mdl, tokenizer=tok,
        max_new_tokens=MAX_NEW_TOKENS, temperature=0.0,
        do_sample=False, device_map="auto",
    )

    y_true, y_pred, raw_outputs = [], [], []

    print(f"Evaluating {len(samples)} samples...")
    for i, s in enumerate(samples):
        prompt    = make_prompt(s.history, s.history_speakers, s.history_emotions, s.target_speaker)
        out       = gen(prompt)[0]["generated_text"]
        generated = out[len(prompt):].strip()
        emotion   = parse_emotion(generated)
        pred_id   = pred_to_id(emotion)

        y_true.append(s.target_emotion_id)
        y_pred.append(pred_id)
        raw_outputs.append(generated)

        if (i+1) % 100 == 0:
            print(f"  [{i+1}/{len(samples)}]")

    return evaluate(y_true, y_pred, "Baseline 4: Direct RL Discrete",
                    raw_outputs=raw_outputs, save_path=save_path)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline 4: Direct RL Discrete")
    parser.add_argument("--mode",       required=True,
                        choices=["inspect", "train", "eval"])
    parser.add_argument("--data_path",  default=None)
    parser.add_argument("--model_path", default=None,
                        help="Path to trained model (eval mode). "
                             "Defaults to ./output/b4/final")
    parser.add_argument("--base_model", default=BASE_MODEL)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--loader",     default="pkl", choices=["pkl", "json"])
    parser.add_argument("--max_train",  type=int, default=0,
                        help="Cap number of training samples (0=all). Use for debug.")
    parser.add_argument("--grpo_steps", type=int, default=0,
                        help="Override GRPO steps (0=default 300). Use small for debug.")
    parser.add_argument("--grpo_g", type=int, default=0, help="GRPO candidates per prompt (0=default 2; use 6 if approved)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split",      default="test",
                        choices=["train", "dev", "test"])
    parser.add_argument("--save_path",  default=None,
                        help="Save predictions to this JSON file")
    args = parser.parse_args()

    if args.mode == "inspect":
        if not args.data_path:
            parser.error("--data_path required for inspect")
        inspect_pkl(args.data_path)

    elif args.mode == "train":
        if not args.data_path:
            parser.error("--data_path required for train")
        train(args.data_path, args.base_model, args.output_dir, args.loader,
              max_train=args.max_train, grpo_steps=args.grpo_steps,
              grpo_g=args.grpo_g, seed=args.seed)

    elif args.mode == "eval":
        if not args.data_path:
            parser.error("--data_path required for eval")
        model_path = args.model_path or os.path.join(args.output_dir, "final")
        run_eval(model_path, args.data_path, args.loader,
                 args.split, args.save_path)
