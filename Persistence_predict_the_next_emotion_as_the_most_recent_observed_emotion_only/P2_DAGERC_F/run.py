"""
P2 — DAG-ERC-F (causal DAG-ERC adaptation for emotion forecasting)
====================================================================
This is a CAUSAL ADAPTATION inspired by Shen, Wu, Yang, Quan, "Directed
Acyclic Graph Network for Conversational Emotion Recognition," ACL-IJCNLP
2021 (arXiv:2105.12907, code: github.com/shenwzh3/DAG-ERC) -- it is NOT an
exact reproduction. Call it "DAG-ERC-F" or "DAG-ERC causal adaptation" in
any writeup, never "DAG-ERC" unqualified.

STRICT CAUSALITY / NO-LEAKAGE GUARANTEE:
  - The graph contains ONLY historical turns (indices 0..t-1 of the target
    sample). The target turn t is never a real graph node with real content.
  - A virtual target node is appended with a learned "unknown-utterance"
    embedding standing in for the missing text -- never the real target
    utterance's text or its embedding.
  - The only thing about turn t that IS available (matching the task
    definition) is the identity of the next speaker, used to pick the
    target node's speaker-edge-typing exactly as the paper's own
    "interlocutor identity is known, content is not" framing requires.
  - Nothing about turns > t is ever read (no future context at all, not
    just the immediate next turn).

See README.md "Deviations from the original paper" for the full list of
adaptations and why each was made -- do not treat this as DAG-ERC itself.

Run:
  # Stage 1: cache frozen utterance features (once per dataset)
  python run.py --mode cache --data_path /path/to/iemocap.pkl --config config.yaml

  # Stage 2: train the graph net
  python run.py --mode train --data_path /path/to/iemocap.pkl --config config.yaml

  # Stage 3: evaluate + save predictions (P0-compatible format)
  python run.py --mode eval --data_path /path/to/iemocap.pkl --config config.yaml \
      --model_path outputs/best.pt --save_path outputs/predictions.json
"""
import os, sys, argparse, random, json, subprocess, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from shared.iemocap_utils import (
    load_iemocap_pkl, EMOTION_LABELS, LABEL2ID, NUM_LABELS, evaluate,
)


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
        return "n/a (not a git repo or git unavailable)"


# ---- Stage 1: frozen utterance-feature cache --------------------------------
def conv_id_and_idx(dialogue_id):
    """dialogue_id is f'{vid}_t{t}'. Recover (vid, t)."""
    vid, _, t = dialogue_id.rpartition("_t")
    return vid, int(t)


def build_feature_cache(splits, encoder_name, device, batch_size=32):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(encoder_name)
    enc = AutoModel.from_pretrained(encoder_name).to(device).eval()

    seen = {}
    for split in splits.values():
        for s in split:
            vid, _ = conv_id_and_idx(s.dialogue_id)
            for i, utt_text in enumerate(s.history):
                seen[(vid, i)] = utt_text
            # s.target_emotion's TEXT is never read here -- only s.history.
            # This loop structurally cannot see the target utterance's text
            # because Sample objects don't carry it at all (see shared/
            # iemocap_utils.py Sample dataclass: only history/history_speakers/
            # history_emotions/target_speaker/target_emotion(_id) exist).

    keys = list(seen.keys())
    texts = [seen[k] for k in keys]
    print(f"Encoding {len(texts)} unique utterances with frozen {encoder_name}...")
    feats = {}
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_keys = keys[i:i + batch_size]
            batch_txt = texts[i:i + batch_size]
            t = tok(batch_txt, return_tensors="pt", padding=True, truncation=True,
                    max_length=64).to(device)
            out = enc(**t).last_hidden_state
            mask = t["attention_mask"].unsqueeze(-1).float()
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = pooled.cpu()
            for k, v in zip(batch_keys, pooled):
                feats[k] = v
            if (i // batch_size) % 20 == 0:
                print(f"  [{min(i+batch_size,len(texts))}/{len(texts)}]", flush=True)
    dim = next(iter(feats.values())).shape[0]
    print(f"Done. Feature dim={dim}, cached {len(feats)} utterances.")
    return feats, dim


# ---- Stage 2: DAG relational graph network -----------------------------------
class DAGLayer(nn.Module):
    def __init__(self, hidden, window):
        super().__init__()
        self.window = window
        self.edge_type = nn.Embedding(2, hidden)   # 0=same-speaker, 1=diff-speaker
        self.attn_q = nn.Linear(hidden, hidden)
        self.attn_k = nn.Linear(hidden, hidden)
        self.gru = nn.GRUCell(hidden, hidden)

    def forward(self, node_states, edge_same_speaker_flags):
        n = node_states.size(0)
        out = []  # list, not in-place tensor writes -- see DEVIATIONS.md bug note
        for i in range(n):
            lo = max(0, i - self.window)
            if i == lo:
                ctx = torch.zeros_like(node_states[i])
            else:
                preds = torch.stack(out[lo:i])
                flags = torch.tensor(edge_same_speaker_flags[i], device=node_states.device)
                etype = self.edge_type(flags)
                keys = self.attn_k(preds) + etype
                q = self.attn_q(node_states[i]).unsqueeze(0)
                scores = (q @ keys.t()).squeeze(0) / (keys.size(-1) ** 0.5)
                w = F.softmax(scores, dim=0)
                ctx = (w.unsqueeze(-1) * preds).sum(0)
            updated = self.gru(ctx.unsqueeze(0), node_states[i].unsqueeze(0)).squeeze(0)
            out.append(updated)
        return torch.stack(out)


class DAGERC_F(nn.Module):
    def __init__(self, feat_dim, hidden=300, n_layers=4, window=10, dropout=0.2,
                 n_classes=NUM_LABELS):
        super().__init__()
        self.in_proj = nn.Linear(feat_dim, hidden)
        self.unknown_utt = nn.Parameter(torch.randn(feat_dim) * 0.02)
        self.layers = nn.ModuleList([DAGLayer(hidden, window) for _ in range(n_layers)])
        self.out = nn.Linear(hidden, n_classes)
        self.emo_embed = nn.Embedding(NUM_LABELS + 1, hidden)
        self.dropout = nn.Dropout(dropout)
        self.window = window

    def forward(self, feats, speakers, target_speaker, history_emotion_ids):
        device = feats.device
        n = feats.size(0)
        all_speakers = speakers + [target_speaker]
        node_in = torch.cat([feats, self.unknown_utt.unsqueeze(0).to(device)], dim=0)
        node_h = self.in_proj(node_in)
        emo_ids = history_emotion_ids + [NUM_LABELS]
        node_h = node_h + self.emo_embed(torch.tensor(emo_ids, device=device))
        node_h = self.dropout(node_h)

        edge_flags = []
        for i in range(n + 1):
            lo = max(0, i - self.window)
            flags = [1 if all_speakers[j] != all_speakers[i] else 0 for j in range(lo, i)]
            edge_flags.append(flags)

        for layer in self.layers:
            node_h = layer(node_h, edge_flags)

        return self.out(node_h[-1])


def sample_to_tensors(s, cache, device):
    vid, _ = conv_id_and_idx(s.dialogue_id)
    feats = torch.stack([cache[(vid, i)] for i in range(len(s.history))]).to(device)
    emo_ids = [LABEL2ID[e] if e in LABEL2ID else NUM_LABELS for e in s.history_emotions]
    return feats, s.history_speakers, s.target_speaker, emo_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["cache", "train", "eval"])
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--cache_dir", default="outputs/cache")
    ap.add_argument("--output_dir", default="outputs")
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--save_path", default="outputs/predictions.json")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    # CLI overrides for anything in config.yaml
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--learning_rate", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--gnn_layers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max_train_samples", type=int, default=0,
                    help="0 = full train split; >0 subsamples for quick/Colab-scale runs. Never applied to test.")
    ap.add_argument("--max_dev_samples", type=int, default=0,
                    help="0 = full dev split; >0 subsamples for quick/Colab-scale runs. Never applied to test.")
    args = ap.parse_args()

    cfg = load_config(args.config, {
        "epochs": args.epochs, "learning_rate": args.learning_rate,
        "batch_size": args.batch_size, "gnn_layers": args.gnn_layers, "seed": args.seed,
    })

    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = load_iemocap_pkl(args.data_path)
    if args.max_train_samples:
        splits["train"] = splits["train"][:args.max_train_samples]
    if args.max_dev_samples:
        splits["dev"] = splits["dev"][:args.max_dev_samples]
    print(f"[data] train={len(splits['train'])} dev={len(splits['dev'])} test={len(splits['test'])} "
          f"(test is NEVER subsampled)")

    if args.mode == "cache":
        feats, dim = build_feature_cache(splits, cfg["encoder"], device)
        os.makedirs(args.cache_dir, exist_ok=True)
        torch.save({"feats": feats, "dim": dim, "encoder": cfg["encoder"]},
                   os.path.join(args.cache_dir, "features.pt"))
        print(f"Cached -> {args.cache_dir}/features.pt")
        return

    cache_path = os.path.join(args.cache_dir, "features.pt")
    cache_blob = torch.load(cache_path)
    cached_encoder = cache_blob.get("encoder")
    if cached_encoder != cfg["encoder"]:
        raise RuntimeError(f"Feature cache encoder mismatch: cache={cached_encoder!r}, config={cfg['encoder']!r}. Rebuild the cache.")
    cache, feat_dim = cache_blob["feats"], cache_blob["dim"]

    if args.mode == "train":
        model = DAGERC_F(feat_dim, hidden=cfg["hidden"], n_layers=cfg["gnn_layers"],
                         window=cfg["window"], dropout=cfg["dropout"]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=1e-5)
        ce = nn.CrossEntropyLoss()

        train_s, dev_s = splits["train"], splits["dev"]
        bsz = cfg["batch_size"]
        best_dev, best_state = -1, None

        for ep in range(cfg["epochs"]):
            model.train()
            idx = list(range(len(train_s))); random.shuffle(idx)
            total_loss = 0.0
            opt.zero_grad()
            for step, j in enumerate(idx):
                s = train_s[j]
                feats, spk, tgt_spk, emo_ids = sample_to_tensors(s, cache, device)
                logits = model(feats, spk, tgt_spk, emo_ids)
                # Gradient-accumulation batching: DAG-ERC-F processes one
                # conversation graph at a time (variable-length, variable
                # edge structure per sample -- no padded-tensor batching in
                # this dependency-light version, see README "Deviations").
                # `batch_size` from config groups this many samples' losses
                # into one optimizer step, which is the standard way to
                # realize a given effective batch size without padding.
                loss = ce(logits.unsqueeze(0), torch.tensor([s.target_emotion_id], device=device)) / bsz
                loss.backward()
                total_loss += loss.item() * bsz
                if (step + 1) % bsz == 0 or (step + 1) == len(idx):
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step(); opt.zero_grad()

            model.eval(); preds, gold = [], []
            with torch.no_grad():
                for s in dev_s:
                    feats, spk, tgt_spk, emo_ids = sample_to_tensors(s, cache, device)
                    logits = model(feats, spk, tgt_spk, emo_ids)
                    preds.append(logits.argmax().item()); gold.append(s.target_emotion_id)
            from sklearn.metrics import f1_score
            dev_f1 = f1_score(gold, preds, average="weighted",
                              labels=list(range(NUM_LABELS)), zero_division=0)
            print(f"epoch {ep+1}/{cfg['epochs']}  train_loss={total_loss/len(idx):.4f}  dev_wF1={dev_f1:.4f}")
            if dev_f1 > best_dev:
                best_dev = dev_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        os.makedirs(args.output_dir, exist_ok=True)
        torch.save({"state_dict": best_state, "feat_dim": feat_dim, "config": cfg},
                   os.path.join(args.output_dir, "best.pt"))
        with open(os.path.join(args.output_dir, "run_metadata.json"), "w") as f:
            json.dump({"command": " ".join(sys.argv), "config": cfg,
                       "best_dev_weighted_f1": round(best_dev, 4),
                       "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
                       "git_commit": git_commit_or_na()}, f, indent=2)
        print(f"Best dev wF1={best_dev:.4f}. Saved -> {args.output_dir}/best.pt")

    elif args.mode == "eval":
        blob = torch.load(args.model_path or os.path.join(args.output_dir, "best.pt"))
        eval_cfg = blob["config"]
        model = DAGERC_F(blob["feat_dim"], hidden=eval_cfg["hidden"], n_layers=eval_cfg["gnn_layers"],
                         window=eval_cfg["window"], dropout=eval_cfg["dropout"]).to(device)
        model.load_state_dict(blob["state_dict"]); model.eval()

        test_s = splits[args.split]
        y_true, y_pred = [], []
        with torch.no_grad():
            for s in test_s:
                feats, spk, tgt_spk, emo_ids = sample_to_tensors(s, cache, device)
                logits = model(feats, spk, tgt_spk, emo_ids)
                y_true.append(s.target_emotion_id); y_pred.append(logits.argmax().item())
        evaluate(y_true, y_pred, "P2 DAG-ERC-F (causal DAG-ERC adaptation)",
                save_path=args.save_path, samples=test_s)


if __name__ == "__main__":
    main()
