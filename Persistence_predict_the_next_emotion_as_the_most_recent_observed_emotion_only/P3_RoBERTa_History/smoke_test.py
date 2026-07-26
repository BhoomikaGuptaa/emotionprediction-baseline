"""
smoke_test.py — P3 RoBERTa-History
====================================
Tests the data-formatting and dataset/eval-loop logic WITHOUT downloading a
real roberta-base (this sandbox has no network access to huggingface.co).
Uses a minimal mock tokenizer and a controllable mock model instead, so the
things this smoke test actually verifies are the things this file's own
code is responsible for getting right -- not HF's.

Checks:
  1. sample_text() never contains the target utterance's text (leakage
     guard) and does contain the next-speaker marker.
  2. ForecastDataset correctly pads/truncates via the tokenizer interface
     and aligns labels to the right samples.
  3. evaluate_loader() computes the correct weighted F1 against a
     hand-constructed mock model with KNOWN correct/incorrect predictions.

Run:  python smoke_test.py
"""
import os, sys, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run import sample_text, ForecastDataset, evaluate_loader
from shared.iemocap_utils import Sample, LABEL2ID, NUM_LABELS
from torch.utils.data import DataLoader


class MockTokenizer:
    """Whitespace tokenizer producing fixed-length input_ids/attention_mask,
    just enough to exercise ForecastDataset's padding/truncation path."""
    def __call__(self, text, truncation=True, max_length=32, padding="max_length"):
        ids = [hash(w) % 1000 for w in text.split()][:max_length]
        mask = [1] * len(ids)
        while len(ids) < max_length:
            ids.append(0); mask.append(0)
        return {"input_ids": ids, "attention_mask": mask}


class MockModel(torch.nn.Module):
    """Returns logits that argmax to whatever label was baked into
    input_ids[0] (smuggled in via the mock tokenizer's hash) -- lets us
    construct a model with KNOWN correct/incorrect behavior."""
    class Out:
        def __init__(self, logits): self.logits = logits
    def forward(self, input_ids, attention_mask):
        # Force argmax == input_ids[:,0] % NUM_LABELS -- deterministic and checkable.
        b = input_ids.size(0)
        logits = torch.zeros(b, NUM_LABELS)
        target_cols = (input_ids[:, 0].float() % NUM_LABELS).long()
        logits[torch.arange(b), target_cols] = 10.0
        return MockModel.Out(logits)


def main():
    utts = ["hi", "hello", "how are you", "fine thanks", "great"]
    emos = ["neutral", "neutral", "anger", "sadness", "happiness"]
    spks = ["A", "B", "A", "B", "A"]
    s = Sample(dialogue_id="conv1_t4", history=utts[:4], history_speakers=spks[:4],
              history_emotions=emos[:4], target_speaker=spks[4], target_emotion=emos[4],
              target_emotion_id=LABEL2ID[emos[4]])

    text = sample_text(s, max_turns=10)
    assert "how are you" in text and "fine thanks" in text, "history text missing"
    assert "great" not in text, "LEAKAGE: target utterance text found in model input!"
    assert "Next speaker: A" in text, "next-speaker marker missing"
    print("[PASS] sample_text() includes history, next-speaker marker, and never the target text")

    ds = ForecastDataset([s], MockTokenizer(), max_length=16, max_history_turns=10)
    item = ds[0]
    assert item["input_ids"].shape == (16,), f"expected padded length 16, got {item['input_ids'].shape}"
    assert item["label"].item() == LABEL2ID[emos[4]], "label misaligned with sample"
    print("[PASS] ForecastDataset pads correctly and aligns labels to samples")

    # Build 6 samples with KNOWN target labels 0..5, craft input_ids[0] to
    # match label exactly for 4/6 and mismatch for 2/6, check evaluate_loader
    # reports the correct accuracy/F1 rather than trusting it blindly.
    class FixedIdsTokenizer(MockTokenizer):
        def __init__(self, forced_id):
            self.forced_id = forced_id
        def __call__(self, text, truncation=True, max_length=16, padding="max_length"):
            out = super().__call__(text, truncation, max_length, padding)
            out["input_ids"][0] = self.forced_id
            return out

    samples2, correct_ids = [], [0, 1, 2, 3, 4, 5]
    forced = [0, 1, 2, 3, 4, 0]  # last one deliberately WRONG (forced=0, true=5)
    from shared.iemocap_utils import EMOTION_LABELS
    for i in range(6):
        emo = EMOTION_LABELS[i]
        samples2.append(Sample(dialogue_id=f"c_t{i}", history=["x"], history_speakers=["A"],
                               history_emotions=["neutral"], target_speaker="B",
                               target_emotion=emo, target_emotion_id=i))

    class PerSampleDataset(torch.utils.data.Dataset):
        def __init__(self, samples, forced_ids):
            self.samples, self.forced_ids = samples, forced_ids
        def __len__(self): return len(self.samples)
        def __getitem__(self, idx):
            tok = FixedIdsTokenizer(self.forced_ids[idx])
            enc = tok("dummy text", max_length=8)
            return {"input_ids": torch.tensor(enc["input_ids"]),
                    "attention_mask": torch.tensor(enc["attention_mask"]),
                    "label": torch.tensor(self.samples[idx].target_emotion_id)}

    loader = DataLoader(PerSampleDataset(samples2, forced), batch_size=6, shuffle=False)
    wf1, preds, gold = evaluate_loader(MockModel(), loader, torch.device("cpu"))
    assert preds == [0, 1, 2, 3, 4, 0], f"unexpected preds {preds}"
    assert gold == [0, 1, 2, 3, 4, 5], f"unexpected gold {gold}"
    from sklearn.metrics import f1_score
    expected_wf1 = f1_score(gold, preds, average="weighted", labels=list(range(NUM_LABELS)), zero_division=0)
    assert abs(wf1 - expected_wf1) < 1e-9, f"evaluate_loader's F1 ({wf1}) doesn't match sklearn's own computation ({expected_wf1})"
    print(f"[PASS] evaluate_loader() correctly computed weighted F1={wf1:.4f} on a 5/6-correct mock model "
          f"(verified against a hand-checkable prediction pattern, not just 'it ran')")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
