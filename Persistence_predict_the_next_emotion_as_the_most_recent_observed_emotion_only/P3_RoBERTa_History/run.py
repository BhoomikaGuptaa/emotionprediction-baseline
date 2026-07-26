"""
P3 — RoBERTa-History
=====================
A plain supervised roberta-base classifier: flatten the historical dialogue
(text + speaker + gold emotion tags, matching how every other baseline in
this project renders history) plus the next-speaker identity, into one
input string, and classify the NEXT turn's emotion. No target text, ever.

This is the "simplest strong supervised baseline" — distinct from the
prompting-only LLM baselines (B2/B3) and the generative-LLM baseline
(InstructERC/B6): a standard discriminative encoder fine-tuned directly on
this task with cross-entropy loss.

Run:
  python run.py --mode train --data_path /path/to/iemocap.pkl --config config.yaml
  python run.py --mode eval  --data_path /path/to/iemocap.pkl --config config.yaml \
      --model_path outputs/best_model --save_path outputs/predictions.json
"""
import os, sys, argparse, random, json, subprocess, datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from shared.iemocap_utils import load_iemocap_pkl, format_history, NUM_LABELS, evaluate


def load_config(path, overrides):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def git_commit_or_na():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                        stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "n/a"


def sample_text(s, max_turns):
    """History + gold prior emotions + next-speaker marker. NEVER includes
    the target utterance's text -- Sample objects don't carry it at all."""
    hist = format_history(s.history, s.history_speakers, s.history_emotions, max_turns=max_turns)
    return f"{hist}\nNext speaker: {s.target_speaker}"


class ForecastDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length, max_history_turns):
        self.samples = samples
        self.tok = tokenizer
        self.max_length = max_length
        self.max_turns = max_history_turns

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        text = sample_text(s, self.max_turns)
        enc = self.tok(text, truncation=True, max_length=self.max_length, padding="max_length")
        return {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "label": torch.tensor(s.target_emotion_id),
        }


def evaluate_loader(model, loader, device):
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            preds += logits.argmax(-1).cpu().tolist()
            gold += batch["label"].tolist()
    from sklearn.metrics import f1_score
    wf1 = f1_score(gold, preds, average="weighted", labels=list(range(NUM_LABELS)), zero_division=0)
    return wf1, preds, gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["train", "eval"])
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--save_path", default="outputs/predictions.json")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--learning_rate", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max_train_samples", type=int, default=0,
                    help="0 = full train split; >0 subsamples for quick/Colab-scale runs. Never applied to test.")
    ap.add_argument("--max_dev_samples", type=int, default=0,
                    help="0 = full dev split; >0 subsamples for quick/Colab-scale runs. Never applied to test.")
    args = ap.parse_args()

    cfg = load_config(args.config, {
        "epochs": args.epochs, "learning_rate": args.learning_rate,
        "batch_size": args.batch_size, "seed": args.seed,
    })

    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(cfg["encoder"])

    splits = load_iemocap_pkl(args.data_path)
    if args.max_train_samples:
        splits["train"] = splits["train"][:args.max_train_samples]
    if args.max_dev_samples:
        splits["dev"] = splits["dev"][:args.max_dev_samples]
    print(f"[data] train={len(splits['train'])} dev={len(splits['dev'])} test={len(splits['test'])} "
          f"(test is NEVER subsampled)")

    if args.mode == "train":
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg["encoder"], num_labels=NUM_LABELS).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                weight_decay=cfg["weight_decay"])
        ce = nn.CrossEntropyLoss()

        train_ds = ForecastDataset(splits["train"], tok, cfg["max_length"], cfg["max_history_turns"])
        dev_ds = ForecastDataset(splits["dev"], tok, cfg["max_length"], cfg["max_history_turns"])
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
        dev_loader = DataLoader(dev_ds, batch_size=cfg["batch_size"] * 2, shuffle=False)

        accum = cfg["gradient_accumulation_steps"]
        best_dev, best_state = -1, None

        for ep in range(cfg["epochs"]):
            model.train()
            opt.zero_grad()
            total_loss = 0.0
            for step, batch in enumerate(train_loader):
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                logits = model(input_ids=ids, attention_mask=mask).logits
                loss = ce(logits, labels) / accum
                loss.backward()
                total_loss += loss.item() * accum
                if (step + 1) % accum == 0 or (step + 1) == len(train_loader):
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); opt.zero_grad()

            dev_f1, _, _ = evaluate_loader(model, dev_loader, device)
            print(f"epoch {ep+1}/{cfg['epochs']}  train_loss={total_loss/len(train_loader):.4f}  dev_wF1={dev_f1:.4f}")
            if dev_f1 > best_dev:
                best_dev = dev_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        os.makedirs(args.output_dir, exist_ok=True)
        model.load_state_dict(best_state)
        save_dir = os.path.join(args.output_dir, "best_model")
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        with open(os.path.join(args.output_dir, "run_metadata.json"), "w") as f:
            json.dump({"command": " ".join(sys.argv), "config": cfg,
                       "best_dev_weighted_f1": round(best_dev, 4),
                       "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
                       "git_commit": git_commit_or_na()}, f, indent=2)
        print(f"Best dev wF1={best_dev:.4f}. Saved -> {save_dir}")

    elif args.mode == "eval":
        model_path = args.model_path or os.path.join(args.output_dir, "best_model")
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        tok = AutoTokenizer.from_pretrained(model_path)

        test_samples = splits[args.split]
        test_ds = ForecastDataset(test_samples, tok, cfg["max_length"], cfg["max_history_turns"])
        test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"] * 2, shuffle=False)

        _, y_pred, y_true = evaluate_loader(model, test_loader, device)
        evaluate(y_true, y_pred, "P3 RoBERTa-History", save_path=args.save_path, samples=test_samples)


if __name__ == "__main__":
    main()
