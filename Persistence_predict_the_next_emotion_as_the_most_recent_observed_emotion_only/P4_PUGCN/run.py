"""
P4 — PUGCN adapted to canonical IEMOCAP forecasting (v2 — rebuilt from the real paper)
=========================================================================================
Paper: Xie, Liu, Sun, Li, "Pseudo-Utterance-Guided Contrastive Network for
Emotion Forecasting in Conversations," Expert Systems with Applications,
279:127382, 2025. https://doi.org/10.1016/j.eswa.2025.127382

This is v2, rebuilt after reading the actual PDF (v1 was built from an
abstract-only summary and got the mechanics wrong -- see README.md
"What changed from v1" for the full diff). No public code repository exists
for this paper (checked again against the PDF's own Data Availability
statement: "the authors do not have permission to share data" -- no code
link anywhere). This remains an ADAPTATION, not a reproduction: call it
"PUGCN adapted to canonical IEMOCAP forecasting."

ARCHITECTURE (matches the paper's Section 3, Fig. 2):
  1. Interlocutor identity-aware encoding (Section 3.2): each utterance is
     prefixed with its speaker identity, then run through BART in
     "BART-Classification" mode (fed into BOTH encoder and decoder, then
     max-pooled) to get a per-utterance representation h_i.
  2. Conversation-level Transformer (Eq. 6-9): self-attention over
     [h_1, ..., h_{i-1}, h_pseudo] (history representations + the
     pseudo-utterance representation standing in for the missing target)
     produces the final contextualized representation used for
     classification.
  3. Emotion-level contrastive learning (Section 3.3, Eq. 10-12): supervised
     contrastive loss with the paper's specific "detach-and-duplicate" trick
     (doubles a batch of N to 2N by adding a gradient-detached copy of each
     representation) to address small per-class counts in a batch.
  4. Pseudo-utterance generation (Section 3.4, Eq. 13-15): BART in
     "BART-Generation" mode autoregressively generates a pseudo-utterance
     conditioned on history + target speaker identity, trained via
     teacher-forced cross-entropy AGAINST THE REAL TARGET UTTERANCE'S TEXT.

LEAKAGE ANALYSIS -- read this before trusting any number out of this folder:
  The paper's own PUG loss (Eq. 15) genuinely requires the real target
  utterance's text as a TRAINING supervision signal (teacher forcing). That
  text is never available in this project's Sample objects by design (see
  shared/iemocap_utils.py), so this file adds a SEPARATE, train-split-only
  text-loading path (`load_train_target_texts`) that is structurally
  impossible to call for dev/test (see the `--mode eval` code path, which
  never imports or calls it).

  Critically: that real text is used for EXACTLY ONE THING -- computing
  `pug_loss` (a training-only auxiliary loss on the generator's own
  token-prediction quality). It is NEVER fed into the classification path.
  The representation actually used for classification (`h_pseudo`) is ALWAYS
  built from a FREE-RUNNING GENERATED pseudo-utterance (via `.generate()`,
  no teacher forcing, real text blind) -- identically at train and eval
  time. This is both (a) faithful to the paper's own framing of the
  pseudo-utterance as a stand-in that "replaces" the missing utterance
  throughout the forecasting process, not just during loss computation, and
  (b) the only version of this that is actually safe under this project's
  leakage requirement. An earlier draft of this file (during this session)
  briefly considered reusing teacher-forced decoder hidden states as
  `h_pseudo` for speed -- that would have been a real, silent leakage bug
  (the classifier's input would have been built from real target text ids
  during training) and was rejected before being implemented; see README.md
  "Design decisions" for the full reasoning.

Run:
  python run.py --mode train --data_path /path/to/iemocap.pkl --config config.yaml
  python run.py --mode eval  --data_path /path/to/iemocap.pkl --config config.yaml \
      --model_path outputs/best.pt --save_path outputs/predictions.json
"""
import os, sys, argparse, random, json, subprocess, datetime, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from shared.iemocap_utils import load_iemocap_pkl, EMOTION_LABELS, LABEL2ID, NUM_LABELS, evaluate


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


# ---- TRAIN-ONLY target text loader (structurally separate from Sample) ------
def load_train_target_texts(pkl_path):
    """Return {sample.dialogue_id: target_text} for canonical TRAIN samples only.

    Supports the same three raw formats as shared.load_iemocap_pkl, including
    the standard DialogueRNN tuple. The function independently reconstructs
    the deterministic 100/20/31 dialogue split and exposes target text only for
    train dialogue IDs. Dev/test target text is never returned.
    """
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    texts = {}
    if isinstance(raw, dict) and "train" in raw:
        for dlg in raw["train"]:
            utts = dlg.get("utterance", dlg.get("utterances", []))
            vid = dlg.get("conv_id", dlg.get("vid"))
            if vid is None:
                continue
            for t in range(1, len(utts)):
                texts[f"{vid}_t{t}"] = str(utts[t])
        return texts
    if isinstance(raw, (list, tuple)) and len(raw) >= 9:
        sentences, train_vids = raw[6], list(raw[7])
        train_sorted = sorted(train_vids)
        n_val = 20 if len(train_sorted) > 20 else (max(1, int(0.1 * len(train_sorted))) if len(train_sorted) > 10 else 0)
        val = set(train_sorted[-n_val:]) if n_val else set()
        train_ids = set(train_sorted) - val
        for vid in train_ids:
            utts = sentences.get(vid, [])
            for t in range(1, len(utts)):
                texts[f"{vid}_t{t}"] = str(utts[t])
        return texts
    if isinstance(raw, dict) and ("videoSentence" in raw or "trainVid" in raw):
        sentences = raw.get("videoSentence", {})
        train_sorted = sorted(list(raw.get("trainVid", [])))
        n_val = 20 if len(train_sorted) > 20 else (max(1, int(0.1 * len(train_sorted))) if len(train_sorted) > 10 else 0)
        val = set(train_sorted[-n_val:]) if n_val else set()
        train_ids = set(train_sorted) - val
        for vid in train_ids:
            utts = sentences.get(vid, [])
            for t in range(1, len(utts)):
                texts[f"{vid}_t{t}"] = str(utts[t])
        return texts
    raise ValueError(f"Unsupported pickle format for target-text extraction: {type(raw)}")


# ---- Supervised contrastive loss with the paper's detach-duplicate trick ----
def supcon_loss_doubled(H_bar, labels, temp):
    """Implements Eq. 10-12 exactly: given N representations with labels,
    build a 2N batch by appending a gradient-DETACHED copy of each
    representation (Eq. 10: H = [H_bar, H_hat]), then compute supervised
    contrastive loss over all 2N. This directly addresses small per-class
    counts within a batch (the paper's stated motivation) by guaranteeing
    every sample has at least one positive (its own detached copy) even if
    no other batch member shares its label."""
    N = H_bar.size(0)
    H_hat = H_bar.detach().clone()
    H = torch.cat([H_bar, H_hat], dim=0)          # (2N, d)
    all_labels = torch.cat([labels, labels], dim=0)  # (2N,)

    H = F.normalize(H, dim=-1)
    sim = H @ H.t() / temp
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(2 * N, device=H.device)
    exp = torch.exp(sim) * (1 - self_mask)
    log_prob = sim - torch.log(exp.sum(1, keepdim=True) + 1e-9)

    same = (all_labels.unsqueeze(0) == all_labels.unsqueeze(1)).float() * (1 - self_mask)
    denom = same.sum(1)
    per_sample_loss = -(same * log_prob).sum(1) / denom.clamp(min=1)
    valid = denom > 0
    return per_sample_loss[valid].mean() if valid.any() else H_bar.sum() * 0.0


# ---- Conversation-level Transformer + classifier (Eq. 6-9, 16-17) -----------
class ConversationEncoder(nn.Module):
    """Self-attention over [h_1, ..., h_{i-1}, h_pseudo]. Deliberately kept
    as a standalone module operating on already-computed utterance
    representations (not raw text) so it can be unit-tested without BART --
    see smoke_test.py."""
    def __init__(self, d_model, n_layers, n_heads, n_classes, dropout):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                           dropout=dropout, batch_first=True,
                                           dim_feedforward=d_model * 2)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_seq):
        """h_seq: (n, d) representations in order, LAST position = target
        (the pseudo-utterance's representation). Returns (logits, h_bar_target)."""
        ctx = self.transformer(h_seq.unsqueeze(0)).squeeze(0)  # (n, d)
        h_bar = ctx[-1]
        logits = self.classifier(self.dropout(h_bar))
        return logits, h_bar


# ---- BART wrapper: classification mode + generation mode --------------------
class PUGCNReal(nn.Module):
    def __init__(self, bart_name, n_classes, dropout, convo_layers, convo_heads):
        super().__init__()
        from transformers import BartForConditionalGeneration
        self.bart = BartForConditionalGeneration.from_pretrained(bart_name)
        d = self.bart.config.d_model
        self.convo = ConversationEncoder(d, convo_layers, convo_heads, n_classes, dropout)
        self.d_model = d

    def classify_encode(self, input_ids, attention_mask):
        """BART-Classification mode (Eq. 4-5): feed into BOTH encoder and
        decoder, then max-pool the decoder's output sequence."""
        enc = self.bart.model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        dec = self.bart.model.decoder(input_ids=input_ids, attention_mask=attention_mask,
                                      encoder_hidden_states=enc,
                                      encoder_attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).bool()
        dec = dec.masked_fill(~mask, float("-inf"))
        pooled = dec.max(dim=1).values
        return pooled  # (batch, d)

    def pug_loss(self, gen_input_ids, gen_attn, target_ids):
        """Eq. 13-15: teacher-forced seq2seq cross-entropy against the REAL
        target utterance's tokens. TRAIN-ONLY -- caller must never invoke
        this without genuine train-split target_ids."""
        labels = target_ids.clone()
        labels[labels == self.bart.config.pad_token_id] = -100
        out = self.bart(input_ids=gen_input_ids, attention_mask=gen_attn, labels=labels)
        return out.loss

    @torch.no_grad()
    def generate_pseudo_ids(self, gen_input_ids, gen_attn, max_new_tokens):
        """FREE-RUNNING generation -- never sees real target text. This is
        the ONLY path that feeds into classification (train AND eval)."""
        return self.bart.generate(input_ids=gen_input_ids, attention_mask=gen_attn,
                                  max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)


def build_history_strings(history, history_speakers):
    return [f"{spk}: {utt}" for spk, utt in zip(history_speakers, history)]


def forward_sample(model, tokenizer, s, cfg, device, real_target_text=None):
    """One sample's full forward pass. `real_target_text` is None at eval
    (structurally -- eval's call site never has it, see main()), and when
    provided at train time is used ONLY inside pug_loss_val below, never
    threaded into the classify_encode calls that produce h_pseudo."""
    hist_strs = build_history_strings(s.history, s.history_speakers)

    # 1. Classification-mode encode every REAL history utterance.
    if hist_strs:
        enc_in = tokenizer(hist_strs, return_tensors="pt", padding=True, truncation=True,
                           max_length=cfg["max_utt_length"]).to(device)
        h_hist = model.classify_encode(enc_in["input_ids"], enc_in["attention_mask"])
    else:
        h_hist = torch.zeros(0, model.d_model, device=device)

    # 2. Generation-mode: encoder input = history concat + target speaker marker.
    hist_concat = " ".join(hist_strs)
    gen_prompt = f"{hist_concat} {s.target_speaker}:"
    gen_in = tokenizer(gen_prompt, return_tensors="pt", truncation=True,
                       max_length=cfg.get("max_history_tokens", cfg.get("max_utt_length", 48) * 4)).to(device)

    pug_loss_val = None
    if real_target_text is not None:
        tgt_enc = tokenizer(real_target_text, return_tensors="pt", truncation=True,
                            max_length=cfg["max_utt_length"]).to(device)
        pug_loss_val = model.pug_loss(gen_in["input_ids"], gen_in["attention_mask"], tgt_enc["input_ids"])

    # 3. FREE-RUNNING pseudo-utterance generation (real-text-blind, always).
    gen_ids = model.generate_pseudo_ids(gen_in["input_ids"], gen_in["attention_mask"],
                                        cfg["gen_max_new_tokens"])
    pseudo_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    pseudo_str = f"{s.target_speaker}: {pseudo_text}"
    pseudo_enc = tokenizer([pseudo_str], return_tensors="pt", padding=True, truncation=True,
                           max_length=cfg["max_utt_length"]).to(device)
    h_pseudo = model.classify_encode(pseudo_enc["input_ids"], pseudo_enc["attention_mask"])  # (1, d)

    # 4. Conversation-level Transformer over [h_hist..., h_pseudo].
    h_seq = torch.cat([h_hist, h_pseudo], dim=0)
    logits, h_bar = model.convo(h_seq)
    return logits, h_bar, pug_loss_val


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
                    help="0 = full train split; >0 subsamples for quick/Colab-scale runs. Never applied to test. "
                         "STRONGLY recommended on Colab -- P4 runs .generate() every training step, full IEMOCAP "
                         "train (~5-6k samples) will not fit a Colab session.")
    ap.add_argument("--max_dev_samples", type=int, default=0,
                    help="0 = full dev split; >0 subsamples for quick/Colab-scale runs. Never applied to test.")
    args = ap.parse_args()

    cfg = load_config(args.config, {
        "epochs": args.epochs, "learning_rate": args.learning_rate,
        "batch_size": args.batch_size, "seed": args.seed,
    })

    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["encoder"])
    splits = load_iemocap_pkl(args.data_path)
    if args.max_train_samples:
        splits["train"] = splits["train"][:args.max_train_samples]
    if args.max_dev_samples:
        splits["dev"] = splits["dev"][:args.max_dev_samples]
    print(f"[data] train={len(splits['train'])} dev={len(splits['dev'])} test={len(splits['test'])} "
          f"(test is NEVER subsampled)")

    if args.mode == "train":
        # Train-only target text -- loaded ONCE here, passed only into the
        # train loop below, NEVER into anything eval touches.
        train_target_texts = load_train_target_texts(args.data_path)
        missing = [s.dialogue_id for s in splits["train"] if s.dialogue_id not in train_target_texts]
        if missing:
            raise RuntimeError(f"Missing target text for {len(missing)} canonical train samples; first={missing[:5]}")
        if set(train_target_texts) & {s.dialogue_id for s in splits["dev"] + splits["test"]}:
            raise RuntimeError("Target-text extraction leaked dev/test sample IDs")

        model = PUGCNReal(cfg["encoder"], NUM_LABELS, cfg["dropout"],
                          cfg["convo_transformer_layers"], cfg["convo_transformer_heads"]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-5)
        ce = nn.CrossEntropyLoss()

        train_s, dev_s = splits["train"], splits["dev"]
        bsz = cfg["batch_size"]
        best_dev, best_state = -1, None

        for ep in range(cfg["epochs"]):
            model.train()
            idx = list(range(len(train_s))); random.shuffle(idx)
            total_ce, total_cl, total_pug, n_chunks = 0.0, 0.0, 0.0, 0

            for start in range(0, len(idx), bsz):
                chunk = idx[start:start + bsz]
                opt.zero_grad()
                ce_losses, contexts, labels, pug_losses = [], [], [], []
                for j in chunk:
                    s = train_s[j]
                    real_text = train_target_texts.get(s.dialogue_id)  # None if missing -- pug term just skipped for that sample
                    logits, h_bar, pug_l = forward_sample(model, tokenizer, s, cfg, device, real_target_text=real_text)
                    ce_losses.append(ce(logits.unsqueeze(0), torch.tensor([s.target_emotion_id], device=device)))
                    contexts.append(h_bar); labels.append(s.target_emotion_id)
                    if pug_l is not None:
                        pug_losses.append(pug_l)

                ce_loss = torch.stack(ce_losses).mean()
                cl_loss = supcon_loss_doubled(torch.stack(contexts), torch.tensor(labels, device=device),
                                              cfg["contrast_temp"])
                pug_loss = torch.stack(pug_losses).mean() if pug_losses else torch.tensor(0.0, device=device)

                alpha, beta = cfg["alpha"], cfg["beta"]
                loss = (1 - alpha - beta) * ce_loss + alpha * cl_loss + beta * pug_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                total_ce += ce_loss.item(); total_cl += cl_loss.item()
                total_pug += pug_loss.item() if pug_losses else 0.0
                n_chunks += 1

            model.eval(); preds, gold = [], []
            with torch.no_grad():
                for s in dev_s:
                    logits, _, _ = forward_sample(model, tokenizer, s, cfg, device, real_target_text=None)
                    preds.append(logits.argmax().item()); gold.append(s.target_emotion_id)
            from sklearn.metrics import f1_score
            dev_f1 = f1_score(gold, preds, average="weighted", labels=list(range(NUM_LABELS)), zero_division=0)
            print(f"epoch {ep+1}/{cfg['epochs']}  CE={total_ce/n_chunks:.4f}  "
                  f"CL={total_cl/n_chunks:.4f}  PUG={total_pug/n_chunks:.4f}  dev_wF1={dev_f1:.4f}")
            if dev_f1 > best_dev:
                best_dev = dev_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        os.makedirs(args.output_dir, exist_ok=True)
        torch.save({"state_dict": best_state, "config": cfg}, os.path.join(args.output_dir, "best.pt"))
        with open(os.path.join(args.output_dir, "run_metadata.json"), "w") as f:
            json.dump({"command": " ".join(sys.argv), "config": cfg,
                       "best_dev_weighted_f1": round(best_dev, 4),
                       "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
                       "git_commit": git_commit_or_na()}, f, indent=2)
        print(f"Best dev wF1={best_dev:.4f}. Saved -> {args.output_dir}/best.pt")

    elif args.mode == "eval":
        # NOTE: load_train_target_texts is never called anywhere in this
        # branch -- structurally impossible for real target text to reach
        # this code path, not just "we didn't happen to call it."
        blob = torch.load(args.model_path or os.path.join(args.output_dir, "best.pt"))
        eval_cfg = blob["config"]
        model = PUGCNReal(eval_cfg["encoder"], NUM_LABELS, eval_cfg["dropout"],
                          eval_cfg["convo_transformer_layers"], eval_cfg["convo_transformer_heads"]).to(device)
        model.load_state_dict(blob["state_dict"]); model.eval()

        test_s = splits[args.split]
        y_true, y_pred = [], []
        with torch.no_grad():
            for s in test_s:
                logits, _, _ = forward_sample(model, tokenizer, s, eval_cfg, device, real_target_text=None)
                y_true.append(s.target_emotion_id); y_pred.append(logits.argmax().item())
        evaluate(y_true, y_pred, "P4 PUGCN adapted to canonical IEMOCAP forecasting (v2)",
                save_path=args.save_path, samples=test_s)


if __name__ == "__main__":
    main()
