"""
B7 — EACL (runnable reimplementation, causal next-emotion)
==========================================================
Faithful to Yu et al., "Emotion-Anchored Contrastive Learning Framework for ERC,"
Findings of NAACL 2024. Reference code: github.com/Yu-Fangxu/EACL.

Core pieces reproduced:
  - utterance/context representation from a (SimCSE-)RoBERTa encoder,
  - learnable EMOTION ANCHORS (one vector per class); classification is similarity
    of the representation to the anchors,
  - supervised contrastive loss (SupCon) over the batch to pull same-emotion
    representations together,
  - an anchor-separation term so anchors for different emotions stay apart.

Adapted to forecasting: the encoder reads the labelled causal history (with gold
prior emotions) plus the next-speaker marker, and the target is the NEXT emotion.
No current utterance is seen.

Run:
  python b7_eacl/run.py --data_path data/iemocap.pkl \
      --encoder princeton-nlp/sup-simcse-roberta-large --epochs 8 --save_path results/b7.json
"""
import os, sys, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.iemocap_utils import (
    load_iemocap_pkl, EMOTION_LABELS, NUM_LABELS, evaluate, format_history,
)


def sample_text(s):
    hist = format_history(s.history, s.history_speakers, s.history_emotions, max_turns=10)
    return f"{hist}\nNext speaker: {s.target_speaker}"


class EACL(nn.Module):
    def __init__(self, encoder_name, proj_dim=256, n_classes=NUM_LABELS, temp=0.1):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(encoder_name)
        h = self.enc.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(h, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim))
        self.anchors = nn.Parameter(torch.randn(n_classes, proj_dim))
        self.temp = temp

    def represent(self, input_ids, attn):
        h = self.enc(input_ids=input_ids, attention_mask=attn).last_hidden_state[:, 0]
        z = F.normalize(self.proj(h), dim=-1)
        return z

    def logits(self, z):
        a = F.normalize(self.anchors, dim=-1)
        return z @ a.t() / self.temp                      # cosine sim to anchors


def supcon_loss(z, y, temp=0.1):
    """Supervised contrastive loss (Khosla et al.) over a batch of normalized z."""
    B = z.size(0)
    sim = z @ z.t() / temp
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp = torch.exp(sim) * (1 - torch.eye(B, device=z.device))
    log_prob = sim - torch.log(exp.sum(1, keepdim=True) + 1e-9)
    same = (y.unsqueeze(0) == y.unsqueeze(1)).float() * (1 - torch.eye(B, device=z.device))
    denom = same.sum(1)
    loss = -(same * log_prob).sum(1) / denom.clamp(min=1)
    return loss[denom > 0].mean() if (denom > 0).any() else z.sum() * 0.0


def anchor_sep_loss(anchors):
    a = F.normalize(anchors, dim=-1)
    sim = a @ a.t()
    off = sim - torch.eye(a.size(0), device=a.device)
    return off.clamp(min=0).pow(2).mean()                 # push anchors apart


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--encoder", default="princeton-nlp/sup-simcse-roberta-large")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lambda_scl", type=float, default=0.5)
    ap.add_argument("--lambda_anchor", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_path", default=None)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.encoder)

    splits = load_iemocap_pkl(args.data_path)
    def pack(split):
        return ([sample_text(s) for s in splits[split]],
                [s.target_emotion_id for s in splits[split]])
    tr_x, tr_y = pack("train"); dv_x, dv_y = pack("dev"); te_x, te_y = pack("test")

    model = EACL(args.encoder).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    ce = nn.CrossEntropyLoss()

    def encode_batch(texts):
        t = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        return t["input_ids"].to(device), t["attention_mask"].to(device)

    def evaluate_split(X, Y):
        model.eval(); preds = []
        with torch.no_grad():
            for i in range(0, len(X), args.batch_size):
                ids, am = encode_batch(X[i:i+args.batch_size])
                z = model.represent(ids, am)
                preds += model.logits(z).argmax(-1).cpu().tolist()
        from sklearn.metrics import f1_score
        return f1_score(Y, preds, average="weighted", labels=list(range(NUM_LABELS)),
                        zero_division=0), preds

    idx = list(range(len(tr_x))); best_dev, best_state = -1, None
    for ep in range(args.epochs):
        model.train(); random.shuffle(idx)
        for i in range(0, len(idx), args.batch_size):
            b = idx[i:i+args.batch_size]
            ids, am = encode_batch([tr_x[j] for j in b])
            y = torch.tensor([tr_y[j] for j in b], device=device)
            z = model.represent(ids, am)
            loss = ce(model.logits(z), y) \
                 + args.lambda_scl * supcon_loss(z, y) \
                 + args.lambda_anchor * anchor_sep_loss(model.anchors)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        dev_f1, _ = evaluate_split(dv_x, dv_y)
        if dev_f1 > best_dev:
            best_dev = dev_f1; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"  epoch {ep+1}: dev wF1={dev_f1:.4f} (best {best_dev:.4f})")

    if best_state: model.load_state_dict(best_state)
    _, preds = evaluate_split(te_x, te_y)
    evaluate(te_y, preds, f"B7 EACL (causal, {args.encoder.split('/')[-1]}, seed{args.seed})",
             save_path=args.save_path)


if __name__ == "__main__":
    main()
