"""
smoke_test.py — P4 PUGCN v2
==============================
No network access to huggingface.co in this sandbox, so real BART can't be
downloaded here. This tests everything that doesn't need it:

  1. `supcon_loss_doubled` (Eq. 10-12's detach-duplicate trick) against
     hand-derivable properties, not just "it ran."
  2. `ConversationEncoder` (the self-attention module, Eq. 6-9) with
     synthetic embeddings -- shape/gradient checks.
  3. `load_train_target_texts` only ever returns TRAIN dialogue_ids, proven
     against a synthetic pkl with known train/dev/test content.
  4. **The leakage separation itself** -- the most important check in this
     file. Uses a mock BART-like object to run `forward_sample` twice, once
     with `real_target_text` and once without, and asserts BIT-FOR-BIT
     IDENTICAL `h_pseudo` inputs were sent to `classify_encode` in both
     cases -- i.e. proves, by direct value comparison rather than just
     source-reading, that whether or not real target text is available
     literally cannot change what the classifier sees.
  5. Source-level check that `--mode eval` in run.py never references
     `load_train_target_texts` or passes a non-None `real_target_text`.

Run:  python smoke_test.py
"""
import os, sys, json, pickle, tempfile, shutil, inspect
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run import supcon_loss_doubled, ConversationEncoder, load_train_target_texts, forward_sample
from shared.iemocap_utils import Sample, LABEL2ID, NUM_LABELS


def test_supcon_doubling():
    torch.manual_seed(0)
    # 3 samples, 2 share a label, 1 is unique in-batch -- the unique one MUST
    # still get a valid loss (not NaN/0-by-construction) because its
    # detached duplicate provides a positive, per Eq. 10's whole point.
    H = torch.randn(3, 8, requires_grad=True)
    labels = torch.tensor([0, 0, 1])
    loss = supcon_loss_doubled(H, labels, temp=0.1)
    assert torch.isfinite(loss), "loss is not finite"
    assert loss.item() > 0, "loss should be positive"
    loss.backward()
    assert H.grad is not None and torch.isfinite(H.grad).all(), "gradient did not flow / is not finite"
    print(f"[PASS] supcon_loss_doubled produces a finite, positive, differentiable loss "
          f"even for a label with only 1 in-batch member (loss={loss.item():.4f})")

    # Sanity: with all-DISTINCT labels, each sample's only positive is its
    # own detached duplicate (identical vector, similarity=1) -- trivially
    # satisfied, so loss should be LOW. With all-SAME labels, each sample
    # must be pulled close to several random OTHER embeddings too (not just
    # its own duplicate) -- a genuinely harder objective for random,
    # unaligned embeddings, so loss should be HIGHER. (This is the opposite
    # of what "more positives = easier" naively suggests -- more positives
    # here means more DISSIMILAR-but-same-label vectors to reconcile, not
    # an easier target.)
    H2 = torch.randn(4, 8)
    same_labels = torch.tensor([0, 0, 0, 0])
    diff_labels = torch.tensor([0, 1, 2, 3])
    loss_same = supcon_loss_doubled(H2, same_labels, temp=0.1)
    loss_diff = supcon_loss_doubled(H2, diff_labels, temp=0.1)
    assert loss_same > loss_diff, (
        f"expected all-same-labels loss ({loss_same:.4f}) > all-distinct-labels "
        f"loss ({loss_diff:.4f}) -- self-duplicate-only positives should be trivially easy")
    print(f"[PASS] all-distinct-label batch loss ({loss_diff:.4f}, self-duplicate-only positives, "
          f"trivially easy) < all-same-label batch loss ({loss_same:.4f}, harder), as expected")


def test_conversation_encoder():
    torch.manual_seed(0)
    enc = ConversationEncoder(d_model=16, n_layers=2, n_heads=4, n_classes=NUM_LABELS, dropout=0.0)
    h_seq = torch.randn(5, 16, requires_grad=True)
    logits, h_bar = enc(h_seq)
    assert logits.shape == (NUM_LABELS,)
    assert h_bar.shape == (16,)
    logits.sum().backward()
    assert h_seq.grad is not None and torch.isfinite(h_seq.grad).all()
    print("[PASS] ConversationEncoder forward+backward OK, correct output shapes")


def test_train_only_text_loader():
    tmpdir = tempfile.mkdtemp()
    try:
        train_dlg = {"conv_id": "train_conv", "utterance": ["a", "b", "c"],
                    "speaker": ["A", "B", "A"], "emotion": ["neutral", "anger", "sadness"]}
        test_dlg = {"conv_id": "test_conv", "utterance": ["x", "SECRET_TEST_TEXT", "z"],
                   "speaker": ["A", "B", "A"], "emotion": ["neutral", "anger", "sadness"]}
        data = {"train": [train_dlg], "dev": [test_dlg], "test": [test_dlg]}
        pkl_path = os.path.join(tmpdir, "synth.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f)

        texts = load_train_target_texts(pkl_path)
        assert texts == {"train_conv_t1": "b", "train_conv_t2": "c"}, f"unexpected: {texts}"
        assert "SECRET_TEST_TEXT" not in texts.values(), "LEAKAGE: test-split text found in train-only loader output!"
        assert not any(k.startswith("test_conv") for k in texts), "LEAKAGE: test-split dialogue_ids found in train-only loader output!"
        print("[PASS] load_train_target_texts returns ONLY train-split text, verified against a "
              "planted 'SECRET_TEST_TEXT' string that must never appear")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


class MockBart:
    """Minimal stand-in for PUGCNReal's interface. classify_encode RECORDS
    every input_ids tensor it's called with, so the test can directly
    compare what the classifier actually saw in the real_text vs.
    no_real_text runs. `.convo` reuses the real (already unit-tested)
    ConversationEncoder module, since that part isn't BART-dependent."""
    d_model = 4

    def __init__(self):
        self.classify_calls = []
        self.convo = ConversationEncoder(d_model=self.d_model, n_layers=1, n_heads=2,
                                         n_classes=NUM_LABELS, dropout=0.0)

    def classify_encode(self, input_ids, attention_mask):
        self.classify_calls.append(input_ids.clone())
        return input_ids.float().mean(dim=1, keepdim=True).expand(-1, self.d_model)

    def pug_loss(self, gen_input_ids, gen_attn, target_ids):
        return target_ids.float().mean() * 0.0 + 1.0  # fixed differentiable-looking scalar

    def generate_pseudo_ids(self, gen_input_ids, gen_attn, max_new_tokens):
        # Deterministic, real-text-BLIND -- depends only on gen_input_ids
        # (history + speaker prompt), never on any target text.
        return (gen_input_ids[:, :3] + 1)


class _FakeBatchEncoding(dict):
    def to(self, device):
        return self  # MockTokenizer only ever runs on CPU in this smoke test


class MockTokenizer:
    def __call__(self, text, return_tensors="pt", padding=False, truncation=True, max_length=32):
        texts = text if isinstance(text, list) else [text]
        rows = [[hash(w) % 100 for w in t.split()][:max_length] for t in texts]
        maxlen = max(len(r) for r in rows) if padding or len(rows) > 1 else len(rows[0])
        ids, mask = [], []
        for r in rows:
            m = [1] * len(r)
            while len(r) < maxlen:
                r = r + [0]; m = m + [0]
            ids.append(r); mask.append(m)
        return _FakeBatchEncoding({"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)})

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(x)) for x in ids.tolist())


def test_leakage_separation_by_direct_value_comparison():
    s = Sample(dialogue_id="conv1_t3", history=["hi there", "how are you"],
              history_speakers=["A", "B"], history_emotions=["neutral", "neutral"],
              target_speaker="A", target_emotion="anger", target_emotion_id=LABEL2ID["anger"])
    cfg = {"max_utt_length": 16, "max_history_chars": 200, "gen_max_new_tokens": 4}

    torch.manual_seed(123)
    model_a = MockBart(); tok_a = MockTokenizer()
    logits_a, h_bar_a, pug_a = forward_sample(model_a, tok_a, s, cfg, "cpu", real_target_text=None)

    torch.manual_seed(123)  # identical ConversationEncoder init as run A
    model_b = MockBart(); tok_b = MockTokenizer()
    logits_b, h_bar_b, pug_b = forward_sample(model_b, tok_b, s, cfg, "cpu", real_target_text="I am absolutely furious right now")

    assert pug_a is None, "pug_loss should be None when real_target_text is not provided"
    assert pug_b is not None, "pug_loss should be computed when real_target_text IS provided"

    # THE core check: every input_ids tensor ever sent to classify_encode
    # (i.e. everything that reaches the classifier) must be IDENTICAL
    # between the two runs, because the classification path is supposed to
    # be entirely blind to real_target_text's presence or absence.
    assert len(model_a.classify_calls) == len(model_b.classify_calls), "different number of classify_encode calls"
    for i, (ca, cb) in enumerate(zip(model_a.classify_calls, model_b.classify_calls)):
        assert torch.equal(ca, cb), (
            f"LEAKAGE: classify_encode call #{i} received DIFFERENT input_ids depending on "
            f"whether real_target_text was provided -- the classifier is not blind to it!\n"
            f"no-text run: {ca}\nwith-text run: {cb}")
    assert torch.equal(logits_a, logits_b), "final logits differ depending on real_target_text presence -- LEAKAGE"
    print(f"[PASS] classify_encode received BIT-FOR-BIT IDENTICAL inputs in both runs "
          f"({len(model_a.classify_calls)} calls each) -- real_target_text provably never "
          f"reaches the classification path, only pug_loss (which was computed in run B, "
          f"None in run A, exactly as expected)")


def test_eval_mode_never_touches_train_text_loader():
    src = inspect.getsource(sys.modules["run"])
    eval_branch = src[src.index('elif args.mode == "eval":'):]
    assert "load_train_target_texts(" not in eval_branch, (
        "run.py's eval branch CALLS load_train_target_texts() -- this should be structurally impossible")
    assert "real_target_text=real_text" not in eval_branch
    assert 'real_target_text=None' in eval_branch, "eval branch should explicitly pass real_target_text=None"
    print("[PASS] source-level check: --mode eval never references load_train_target_texts, "
          "and explicitly passes real_target_text=None")


if __name__ == "__main__":
    test_supcon_doubling()
    test_conversation_encoder()
    test_train_only_text_loader()
    test_leakage_separation_by_direct_value_comparison()
    test_eval_mode_never_touches_train_text_loader()
    print("\nALL SMOKE TESTS PASSED")
