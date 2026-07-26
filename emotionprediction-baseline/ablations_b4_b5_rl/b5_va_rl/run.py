"""Paper-ready direct-GRPO ablation for IEMOCAP next-emotion forecasting.

This file is instantiated as either B4 (discrete exact-match reward) or B5
(continuous valence/arousal similarity reward). It intentionally uses:
  * the canonical 100/20/31 dialogue split from shared/iemocap_utils.py;
  * completion-only SFT labels (prompt tokens are -100);
  * strict <emotion>...</emotion> parsing for RL reward;
  * hard-fail target/completion alignment;
  * development-set checkpoint selection before one final test evaluation;
  * resolved runtime metadata rather than module constants.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import inspect
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from shared.iemocap_utils import (  # noqa: E402
    EMOTION_LABELS,
    LABEL2ID,
    NUM_LABELS,
    evaluate,
    inspect_pkl,
    load_iemocap_pkl,
    make_prompt,
    pred_to_id,
)

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
REWARD_KIND = "va"  # discrete | va
VA_SIM: dict[str, dict[str, float]] | None = None
VA_CENTROIDS: dict[str, tuple[float, float]] | None = None
VA_COUNTS: dict[str, int] | None = None
STRICT_EMOTION_RE = re.compile(r"<emotion>\s*([^<]+?)\s*</emotion>", re.I | re.S)
ALIASES = {
    "neutral": "neutral",
    "frustration": "frustration", "frustrated": "frustration",
    "sadness": "sadness", "sad": "sadness",
    "anger": "anger", "angry": "anger",
    "excited": "excited", "excitement": "excited",
    "happiness": "happiness", "happy": "happiness",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def strict_parse_emotion(text: str) -> Optional[str]:
    """Only accept one explicit, valid <emotion>...</emotion> answer."""
    matches = STRICT_EMOTION_RE.findall(text or "")
    if len(matches) != 1:
        return None
    normalized = ALIASES.get(matches[0].strip().lower())
    return normalized if normalized in LABEL2ID else None


def has_strict_format(text: str) -> bool:
    text = text or ""
    return (
        text.count("<think>") == 1
        and text.count("</think>") == 1
        and text.count("<emotion>") == 1
        and text.count("</emotion>") == 1
        and text.index("<think>") < text.index("</think>") < text.index("<emotion>") < text.index("</emotion>")
    )


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, list):
        return " ".join(
            str(item.get("content", "")) for item in completion if isinstance(item, dict)
        )
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def expand_values_strict(values: Any, n_completions: int, field_name: str) -> list[Any]:
    """Align one metadata value per prompt with contiguous GRPO generations.

    TRL may pass either an already-expanded column of length n_completions or one
    value per prompt. Any other shape is a hard API/configuration error.
    """
    if values is None:
        raise RuntimeError(f"GRPO reward did not receive dataset column {field_name!r}.")
    if isinstance(values, str):
        values = [values]
    else:
        values = list(values)
    if not values:
        raise RuntimeError(f"Received an empty {field_name} list.")
    if len(values) == n_completions:
        return values
    if n_completions % len(values) != 0:
        raise RuntimeError(
            f"Cannot align {len(values)} {field_name} value(s) with "
            f"{n_completions} completion(s)."
        )
    generations_per_prompt = n_completions // len(values)
    return [value for value in values for _ in range(generations_per_prompt)]


def expand_targets_strict(targets: Any, n_completions: int) -> list[str]:
    """Expand target labels and verify every expanded value is a valid class."""
    expanded = expand_values_strict(targets, n_completions, "target_emotion")
    unknown = [x for x in expanded if x not in LABEL2ID]
    if unknown:
        raise RuntimeError(f"Unknown target labels passed to reward: {sorted(set(unknown))}")
    return expanded


def reward_value(completion: str, gold: str) -> float:
    pred = strict_parse_emotion(completion)
    reward = 0.2 if has_strict_format(completion) else 0.0
    if pred is None:
        return reward
    reward += 0.2
    if REWARD_KIND == "discrete":
        reward += 0.6 if pred == gold else 0.0
    elif REWARD_KIND == "va":
        if VA_SIM is None:
            raise RuntimeError("V/A similarity matrix has not been initialized.")
        reward += 0.6 * float(VA_SIM[pred][gold])
    else:
        raise RuntimeError(f"Unsupported REWARD_KIND={REWARD_KIND}")
    return float(reward)




EVAL_LINE_RE = re.compile(
    r"^\[(?P<start>[0-9.]+)\s*-\s*(?P<end>[0-9.]+)\]\s+"
    r"(?P<utterance>\S+)\s+(?P<emotion>\S+)\s+\[(?P<dims>[^\]]+)\]"
)
IEMOCAP_CODE_TO_LABEL = {
    "neu": "neutral", "fru": "frustration", "sad": "sadness",
    "ang": "anger", "exc": "excited", "hap": "happiness",
}

def _dialogue_id_from_utterance_id(utterance_id: str) -> str:
    return utterance_id.rsplit("_", 1)[0]

def parse_iemocap_emoevaluation(eval_dir: str) -> list[dict[str, Any]]:
    """Parse original IEMOCAP Session*/dialog/EmoEvaluation/*.txt files.

    Expected dimensional order is [valence, activation, dominance] on the
    original 1--5 scale. Only the six categorical classes used by this project
    are retained.
    """
    root = Path(eval_dir)
    if not root.exists():
        raise FileNotFoundError(f"Emotion-evaluation path does not exist: {root}")
    files = sorted(root.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(
            f"No .txt files found under {root}. Point --emotion_eval_dir to the "
            "IEMOCAP Session*/dialog/EmoEvaluation directory or a parent containing it."
        )
    rows = []
    for file in files:
        with file.open(encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                match = EVAL_LINE_RE.match(line.strip())
                if not match:
                    continue
                code = match.group("emotion").lower()
                label = IEMOCAP_CODE_TO_LABEL.get(code)
                if label is None:
                    continue
                dims = [float(x.strip()) for x in match.group("dims").split(",")]
                if len(dims) < 2:
                    raise RuntimeError(f"Malformed V/A annotation in {file}: {line.strip()}")
                utterance_id = match.group("utterance")
                rows.append({
                    "utterance_id": utterance_id,
                    "dialogue_id": _dialogue_id_from_utterance_id(utterance_id),
                    "emotion": label,
                    "valence_raw": dims[0],
                    "activation_raw": dims[1],
                    "source_file": str(file),
                })
    if not rows:
        raise RuntimeError(f"No six-class dimensional annotations parsed from {root}")
    return rows

def build_train_split_va(eval_dir: str, train_samples) -> tuple[dict[str, tuple[float, float]], dict[str, int], dict[str, dict[str, float]]]:
    """Compute class centroids from original annotations for train dialogues only."""
    train_dialogues = {
        re.sub(r"_t\d+$", "", sample.dialogue_id)
        for sample in train_samples
    }
    rows = [
        row
        for row in parse_iemocap_emoevaluation(eval_dir)
        if row["dialogue_id"] in train_dialogues
    ]
    if not rows:
        raise RuntimeError(
            "No dimensional annotations matched the canonical training dialogue IDs. "
            "Check that the annotation files and iemocap.pkl are from the same IEMOCAP release."
        )
    values = {label: [] for label in EMOTION_LABELS}
    for row in rows:
        # Original IEMOCAP dimensional ratings use a 1--5 scale.
        v = (row["valence_raw"] - 3.0) / 2.0
        a = (row["activation_raw"] - 3.0) / 2.0
        values[row["emotion"]].append((v, a))
    missing = [label for label, xs in values.items() if not xs]
    if missing:
        raise RuntimeError(f"No training V/A annotations found for classes: {missing}")
    centroids = {
        label: (float(np.mean([x[0] for x in xs])), float(np.mean([x[1] for x in xs])))
        for label, xs in values.items()
    }
    counts = {label: len(xs) for label, xs in values.items()}
    distances = {
        (a, b): float(np.linalg.norm(np.asarray(centroids[a]) - np.asarray(centroids[b])))
        for a in EMOTION_LABELS for b in EMOTION_LABELS
    }
    max_distance = max(distances.values())
    if max_distance <= 0:
        raise RuntimeError("Degenerate training-derived V/A centroids")
    similarity = {
        a: {b: max(0.0, 1.0 - distances[(a, b)] / max_distance) for b in EMOTION_LABELS}
        for a in EMOTION_LABELS
    }
    return centroids, counts, similarity

def load_va_centroids_json(path: str) -> tuple[dict[str, tuple[float, float]], dict[str, int], dict[str, dict[str, float]]]:
    payload = json.loads(Path(path).read_text())
    raw_centroids = payload.get("centroids", payload)
    centroids = {label: tuple(map(float, raw_centroids[label])) for label in EMOTION_LABELS}
    counts = {label: int(payload.get("counts", {}).get(label, 0)) for label in EMOTION_LABELS}
    distances = {(a, b): float(np.linalg.norm(np.asarray(centroids[a]) - np.asarray(centroids[b]))) for a in EMOTION_LABELS for b in EMOTION_LABELS}
    max_distance = max(distances.values())
    if max_distance <= 0:
        raise RuntimeError("Degenerate V/A centroids JSON")
    similarity = {a: {b: max(0.0, 1.0 - distances[(a, b)] / max_distance) for b in EMOTION_LABELS} for a in EMOTION_LABELS}
    return centroids, counts, similarity


def make_completion(gold: str) -> str:
    return (
        "<think>\n"
        "Use the dialogue history, prior emotions, and next-speaker identity to forecast the next emotion.\n"
        "</think>\n"
        f"<emotion>{gold}</emotion>"
    )


class CompletionOnlyDataset:
    """Tokenized causal-LM examples with prompt labels masked to -100."""
    def __init__(self, samples, tokenizer, max_length: int):
        self.rows = []
        eos = tokenizer.eos_token or ""
        for sample in samples:
            prompt = make_prompt(
                sample.history,
                sample.history_speakers,
                sample.history_emotions,
                sample.target_speaker,
            ).rstrip() + "\n"
            completion = make_completion(sample.target_emotion) + eos
            prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
            completion_ids = tokenizer(completion, add_special_tokens=False, truncation=False)["input_ids"]
            if len(completion_ids) >= max_length:
                raise ValueError("Completion itself exceeds max_length; increase --max_seq_len.")
            keep_prompt = max_length - len(completion_ids)
            prompt_ids = prompt_ids[-keep_prompt:]
            input_ids = prompt_ids + completion_ids
            labels = [-100] * len(prompt_ids) + completion_ids.copy()
            if not any(label != -100 for label in labels):
                raise RuntimeError("Completion-only masking produced no trainable labels.")
            self.rows.append({"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def completion_only_collator(tokenizer):
    import torch

    def collate(features):
        max_len = max(len(x["input_ids"]) for x in features)
        input_ids, labels, masks = [], [], []
        pad = tokenizer.pad_token_id
        for row in features:
            n = max_len - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [pad] * n)
            labels.append(row["labels"] + [-100] * n)
            masks.append(row["attention_mask"] + [0] * n)
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }
        if not torch.all(batch["labels"][batch["labels"] == -100] == -100):
            raise RuntimeError("Prompt masking invariant failed.")
        return batch

    return collate


def build_lora_config():
    from peft import LoraConfig, TaskType
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )


def generate_prediction(model, tokenizer, prompt: str, max_new_tokens: int) -> tuple[Optional[str], str]:
    import torch
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = out[0, enc["input_ids"].shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return strict_parse_emotion(text), text


def evaluate_model(model, tokenizer, samples, max_new_tokens: int, name: str, save_path: Optional[str] = None):
    y_true, y_pred, raw = [], [], []
    model.eval()
    for idx, sample in enumerate(samples):
        prompt = make_prompt(sample.history, sample.history_speakers, sample.history_emotions, sample.target_speaker)
        pred, text = generate_prediction(model, tokenizer, prompt, max_new_tokens)
        y_true.append(sample.target_emotion_id)
        y_pred.append(pred_to_id(pred))
        raw.append(text)
        if (idx + 1) % 100 == 0:
            print(f"  evaluated {idx + 1}/{len(samples)}")
    return evaluate(y_true, y_pred, name, raw_outputs=raw, save_path=save_path, samples=samples)


def checkpoint_steps(path: Path) -> int:
    try:
        return int(path.name.split("-")[-1])
    except Exception:
        return -1


def load_grpo_checkpoint(base_sft_dir: str, checkpoint_dir: str, dtype):
    """Load merged SFT base and the GRPO LoRA adapter saved by Trainer."""
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(base_sft_dir, torch_dtype=dtype, device_map="auto", attn_implementation="eager")
    adapter_cfg = Path(checkpoint_dir) / "adapter_config.json"
    if not adapter_cfg.exists():
        raise FileNotFoundError(
            f"{checkpoint_dir} does not contain adapter_config.json. "
            "The pinned PEFT/TRL stack must save LoRA adapters at each checkpoint."
        )
    return PeftModel.from_pretrained(base, checkpoint_dir)


def select_best_checkpoint(output_dir: str, sft_merged_dir: str, tokenizer, dev_samples, max_new_tokens: int):
    import torch
    checkpoints = sorted(Path(output_dir).glob("checkpoint-*"), key=checkpoint_steps)
    if not checkpoints:
        raise RuntimeError("No GRPO checkpoint-* directories found; cannot select on dev.")
    records = []
    best = None
    for ckpt in checkpoints:
        print(f"\n[dev-select] evaluating {ckpt}")
        model = load_grpo_checkpoint(sft_merged_dir, str(ckpt), torch.bfloat16)
        metrics = evaluate_model(model, tokenizer, dev_samples, max_new_tokens, f"{REWARD_KIND} dev {ckpt.name}")
        wf1 = float(metrics["weighted_f1"])
        records.append({"checkpoint": str(ckpt), "step": checkpoint_steps(ckpt), "dev_weighted_f1": wf1})
        if best is None or wf1 > best["dev_weighted_f1"]:
            best = records[-1]
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    with open(Path(output_dir) / "dev_checkpoint_scores.json", "w") as f:
        json.dump(records, f, indent=2)
    return best


def package_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in ("torch", "transformers", "trl", "peft", "datasets", "accelerate"):
        try:
            module = __import__(pkg)
            out[pkg] = getattr(module, "__version__", "unknown")
        except Exception:
            out[pkg] = "not-installed"
    return out


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "n/a"


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def train(args):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from peft import get_peft_model
    from trl import GRPOConfig, GRPOTrainer

    global VA_SIM, VA_CENTROIDS, VA_COUNTS
    set_seed(args.seed)
    splits = load_iemocap_pkl(args.data_path)
    train_samples = splits["train"][: args.max_train or None]
    dev_samples = splits["dev"][: args.max_dev or None]
    print(f"train={len(train_samples)} dev={len(dev_samples)} test={len(splits['test'])}")
    if REWARD_KIND == "va":
        if args.va_centroids_json:
            VA_CENTROIDS, VA_COUNTS, VA_SIM = load_va_centroids_json(args.va_centroids_json)
            va_source = {"type": "precomputed_train_only_json", "path": str(Path(args.va_centroids_json).resolve())}
        elif args.emotion_eval_dir:
            # Always derive from the full canonical training split, not a smoke-test subset.
            VA_CENTROIDS, VA_COUNTS, VA_SIM = build_train_split_va(args.emotion_eval_dir, splits["train"])
            va_source = {"type": "IEMOCAP_original_EmoEvaluation_train_dialogues_only", "path": str(Path(args.emotion_eval_dir).resolve())}
        else:
            raise RuntimeError(
                "B5 requires native IEMOCAP dimensional annotations. Provide either "
                "--emotion_eval_dir pointing to Session*/dialog/EmoEvaluation files or "
                "--va_centroids_json containing centroids computed from the canonical training split. "
                "The standard DialogueRNN iemocap.pkl alone does not contain V/A ratings."
            )
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.output_dir) / "va_centroids_train_only.json", "w") as f:
            json.dump({"centroids": VA_CENTROIDS, "counts": VA_COUNTS, "similarity": VA_SIM, "source": va_source, "normalization": "(rating-3)/2 from original 1-5 scale"}, f, indent=2)
        print("[B5] training-derived V/A centroids:", json.dumps(VA_CENTROIDS, indent=2))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    out = Path(args.output_dir)
    sft_adapter_dir = out / "sft_adapter"
    sft_merged_dir = out / "sft_merged"
    out.mkdir(parents=True, exist_ok=True)

    # Stage 1: completion-only SFT.
    sft_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    sft_model = get_peft_model(sft_model, build_lora_config())
    sft_ds = CompletionOnlyDataset(train_samples, tokenizer, args.max_seq_len)
    sanity = completion_only_collator(tokenizer)([sft_ds[0]])
    first_trainable = int((sanity["labels"][0] != -100).nonzero()[0])
    assert torch.all(sanity["labels"][0, :first_trainable] == -100)
    print(f"[SFT] verified completion-only mask; first trainable token index={first_trainable}")

    sft_args = TrainingArguments(
        output_dir=str(out / "sft_trainer"),
        num_train_epochs=args.sft_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.sft_lr,
        bf16=True,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=sft_model, args=sft_args, train_dataset=sft_ds, data_collator=completion_only_collator(tokenizer))
    trainer.train()
    sft_model.save_pretrained(sft_adapter_dir)
    merged = sft_model.merge_and_unload()
    merged.save_pretrained(sft_merged_dir)
    tokenizer.save_pretrained(sft_merged_dir)
    del trainer, sft_model, merged
    torch.cuda.empty_cache()

    # Stage 2: GRPO from the exact same merged SFT base.
    grpo_base = AutoModelForCausalLM.from_pretrained(
        sft_merged_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    grpo_model = get_peft_model(grpo_base, build_lora_config())
    rows = []
    for i, sample in enumerate(train_samples):
        rows.append({
            "prompt": make_prompt(sample.history, sample.history_speakers, sample.history_emotions, sample.target_speaker),
            "target_emotion": sample.target_emotion,
            "sample_id": f"{sample.dialogue_id}__target_{i}",
        })
    grpo_ds = Dataset.from_list(rows)
    debug_printed = {"done": False}

    def reward_fn(completions, prompts=None, **kwargs):
        texts = [completion_to_text(x) for x in completions]
        golds = expand_targets_strict(kwargs.get("target_emotion"), len(texts))
        sample_ids = kwargs.get("sample_id")
        if not debug_printed["done"]:
            expanded_ids = expand_values_strict(sample_ids, len(texts), "sample_id") if sample_ids is not None else ["n/a"] * len(texts)
            for i in range(min(4, len(texts))):
                print(f"[alignment-check] sample={expanded_ids[i]} gold={golds[i]} completion={texts[i][:120]!r}")
            debug_printed["done"] = True
        return [reward_value(text, gold) for text, gold in zip(texts, golds)]

    grpo_kwargs = dict(
        output_dir=str(out),
        max_steps=args.grpo_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.grpo_lr,
        num_generations=args.grpo_g,
        max_completion_length=args.max_new_tokens,
        temperature=1.0,
        beta=args.beta,
        bf16=True,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=max(2, (args.grpo_steps // args.save_steps) + 1),
        report_to="none",
        seed=args.seed,
    )
    accepted = set(inspect.signature(GRPOConfig.__init__).parameters)
    grpo_cfg = GRPOConfig(**{k: v for k, v in grpo_kwargs.items() if k in accepted})
    trainer_kwargs = dict(model=grpo_model, args=grpo_cfg, train_dataset=grpo_ds, reward_funcs=reward_fn)
    trainer_sig = set(inspect.signature(GRPOTrainer.__init__).parameters)
    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig:
        trainer_kwargs["tokenizer"] = tokenizer
    grpo_trainer = GRPOTrainer(**trainer_kwargs)
    grpo_trainer.train()
    del grpo_trainer, grpo_model, grpo_base
    torch.cuda.empty_cache()

    best = select_best_checkpoint(str(out), str(sft_merged_dir), tokenizer, dev_samples, args.max_new_tokens)
    best_model = load_grpo_checkpoint(str(sft_merged_dir), best["checkpoint"], torch.bfloat16)
    final = best_model.merge_and_unload()
    final_dir = out / "best_dev_model"
    final.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    metadata = {
        "baseline": "B4" if REWARD_KIND == "discrete" else "B5",
        "reward_kind": REWARD_KIND,
        "reward": {"format": 0.2, "valid_label": 0.2, "va_similarity": 0.6},
        "base_model": args.base_model,
        "data_sha256": file_sha256(args.data_path),
        "split": "canonical_100_20_31_dialogues",
        "split_sample_counts": {k: len(v) for k, v in splits.items()},
        "seed": args.seed,
        "sft_epochs": args.sft_epochs,
        "sft_lr": args.sft_lr,
        "grpo_steps": args.grpo_steps,
        "grpo_g": args.grpo_g,
        "grpo_lr": args.grpo_lr,
        "beta": args.beta,
        "save_steps": args.save_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.grad_accum,
        "max_seq_len": args.max_seq_len,
        "max_new_tokens": args.max_new_tokens,
        "best_checkpoint": best["checkpoint"],
        "best_dev_weighted_f1": best["dev_weighted_f1"],
        "command": " ".join(sys.argv),
        "git_commit": git_commit(),
        "versions": package_versions(),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "precision": "bf16 LoRA (not 4-bit)",
    }
    if REWARD_KIND == "va":
        metadata["va_centroids"] = VA_CENTROIDS
        metadata["va_counts"] = VA_COUNTS
        metadata["va_similarity_matrix"] = VA_SIM
        metadata["va_source"] = va_source
        metadata["va_note"] = ("Class centroids are computed only from original IEMOCAP dimensional annotations "
                              "belonging to canonical training dialogues. Original 1-5 valence/activation ratings "
                              "are mapped to [-1,1] with (rating-3)/2. The task reward is PURELY "
                              "0.6 * VA_SIM(pred, gold) -- there is no separate exact-match term, by design: this "
                              "isolates 'continuous V/A proximity reward' as the single variable that differs from "
                              "B4's '0.6 * exact_match', matching B4's format/valid-label weights (0.2/0.2) exactly "
                              "so the two are a clean single-variable ablation pair.")
    with open(out / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Selected {best['checkpoint']} by dev weighted F1={best['dev_weighted_f1']:.4f}")
    print(f"Final merged model: {final_dir}")


def run_eval(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    splits = load_iemocap_pkl(args.data_path)
    samples = splits[args.split]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    return evaluate_model(
        model,
        tokenizer,
        samples,
        args.max_new_tokens,
        f"{'B4 discrete' if REWARD_KIND == 'discrete' else 'B5 V/A'} direct GRPO",
        args.save_path,
    )


def self_test():
    global VA_SIM
    VA_SIM = {a: {b: (1.0 if a == b else 0.25) for b in EMOTION_LABELS} for a in EMOTION_LABELS}
    assert strict_parse_emotion("<think>x</think><emotion>anger</emotion>") == "anger"
    assert strict_parse_emotion("I mentioned anger") is None
    assert strict_parse_emotion("<emotion>anger</emotion><emotion>sadness</emotion>") is None
    assert expand_targets_strict(["anger", "sadness"], 4) == ["anger", "anger", "sadness", "sadness"]
    assert expand_values_strict(["sample-A", "sample-B"], 4, "sample_id") == ["sample-A", "sample-A", "sample-B", "sample-B"]
    try:
        expand_targets_strict(None, 2)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Missing targets must hard-fail")
    good = "<think>x</think><emotion>anger</emotion>"
    assert reward_value(good, "anger") > reward_value(good, "sadness")
    print("All local invariant tests passed.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["inspect", "train", "eval", "self_test"])
    p.add_argument("--data_path")
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--output_dir", default=f"outputs/{'b4' if REWARD_KIND == 'discrete' else 'b5'}")
    p.add_argument("--model_path")
    p.add_argument("--save_path", default="outputs/test_predictions.json")
    p.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sft_epochs", type=float, default=2.0)
    p.add_argument("--sft_lr", type=float, default=1e-5)
    p.add_argument("--grpo_steps", type=int, default=300)
    p.add_argument("--grpo_g", type=int, default=2)
    p.add_argument("--grpo_lr", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.001)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--max_train", type=int, default=0)
    p.add_argument("--max_dev", type=int, default=0)
    p.add_argument("--emotion_eval_dir", help="Path containing original IEMOCAP Session*/dialog/EmoEvaluation/*.txt files")
    p.add_argument("--va_centroids_json", help="Alternative precomputed canonical-train-only V/A centroid JSON")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "self_test":
        self_test()
    elif args.mode == "inspect":
        if not args.data_path:
            raise SystemExit("--data_path is required")
        inspect_pkl(args.data_path)
    elif args.mode == "train":
        if not args.data_path:
            raise SystemExit("--data_path is required")
        train(args)
    elif args.mode == "eval":
        if not args.data_path or not args.model_path:
            raise SystemExit("--data_path and --model_path are required")
        run_eval(args)
