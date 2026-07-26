"""
smoke_test.py — P2 DAG-ERC-F
==============================
Tests the model/graph logic directly (DAGERC_F, sample_to_tensors,
conv_id_and_idx) using synthetic in-memory features, WITHOUT needing
network access, a real encoder download, or a GPU. This is deliberate: the
Stage-1 feature-caching step needs a real pretrained encoder (roberta-large
by default) which this sandbox cannot download, so this smoke test isolates
and verifies the part that's actually novel/risky in this reimplementation
-- the DAG relational-attention + GRU graph net and the gradient-accumulation
training loop logic -- rather than the (standard, low-risk) HF encoder call.

Checks:
  1. conv_id_and_idx correctly recovers (vid, turn_index) from dialogue_id.
  2. Forward + backward pass runs cleanly across several epochs with
     monotonically-ish decreasing loss (regression guard against the
     in-place-mutation autograd bug this exact code had during development
     -- see DEVIATIONS.md).
  3. The model can overfit a tiny synthetic conversation (sanity that
     gradients are actually flowing and useful, not just "runs without
     crashing").
  4. Eval-mode (no_grad) forward pass works and predictions are internally
     consistent with what was trained.

Run:  python smoke_test.py
"""
import os, sys, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run import DAGERC_F, conv_id_and_idx, sample_to_tensors
from shared.iemocap_utils import Sample, LABEL2ID


def main():
    assert conv_id_and_idx("Ses01F_impro01_t3") == ("Ses01F_impro01", 3), \
        "conv_id_and_idx failed to parse a realistic dialogue_id"
    assert conv_id_and_idx("conv1_t0") == ("conv1", 0)
    print("[PASS] conv_id_and_idx parses dialogue_id correctly")

    feat_dim = 16
    emos = ["neutral", "neutral", "anger", "neutral", "anger", "sadness"]
    spks = ["A", "B", "A", "B", "A", "B"]
    utts = [f"u{i}" for i in range(6)]
    vid = "conv1"
    torch.manual_seed(0)
    cache = {(vid, i): torch.randn(feat_dim) for i in range(6)}

    samples = []
    for t in range(1, 6):
        samples.append(Sample(
            dialogue_id=f"{vid}_t{t}", history=utts[:t], history_speakers=spks[:t],
            history_emotions=emos[:t], target_speaker=spks[t], target_emotion=emos[t],
            target_emotion_id=LABEL2ID[emos[t]],
        ))

    model = DAGERC_F(feat_dim, hidden=32, n_layers=2, window=10, dropout=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ce = torch.nn.CrossEntropyLoss()

    losses = []
    for epoch in range(40):
        total = 0.0
        for s in samples:
            feats, spk, tgt_spk, emo_ids = sample_to_tensors(s, cache, torch.device("cpu"))
            logits = model(feats, spk, tgt_spk, emo_ids)
            loss = ce(logits.unsqueeze(0), torch.tensor([s.target_emotion_id]))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        losses.append(total / len(samples))
    print(f"[PASS] forward+backward ran cleanly for 40 epochs, final loss: {losses[-1]:.3f} (started at {losses[0]:.3f})")
    assert losses[-1] < losses[0], "loss did not decrease at all -- gradients likely broken"
    print("[PASS] loss decreased over training (gradients are flowing)")

    model.eval()
    correct = 0
    with torch.no_grad():
        for s in samples:
            feats, spk, tgt_spk, emo_ids = sample_to_tensors(s, cache, torch.device("cpu"))
            logits = model(feats, spk, tgt_spk, emo_ids)
            pred = logits.argmax().item()
            correct += int(pred == s.target_emotion_id)
    print(f"[PASS] eval-mode forward pass OK, {correct}/{len(samples)} correct on the tiny overfit set")
    assert correct >= 4, f"expected the model to have overfit at least 4/5 samples, got {correct}/5"

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
