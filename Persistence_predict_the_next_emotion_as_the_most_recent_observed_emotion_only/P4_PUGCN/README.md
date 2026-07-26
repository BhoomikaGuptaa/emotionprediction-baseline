# P4 — PUGCN adapted to canonical IEMOCAP forecasting (v2)

Independent folder. **Rebuilt from the actual paper PDF** — v1 (an earlier
session) was built from an abstract-only summary and got real mechanics
wrong. This version is a genuine architectural rebuild, not a patch.

**Paper:** Xie, Liu, Sun, Li, "Pseudo-Utterance-Guided Contrastive Network
for Emotion Forecasting in Conversations," *Expert Systems with
Applications*, 279:127382, 2025. https://doi.org/10.1016/j.eswa.2025.127382

**No public code exists.** Checked again against the PDF itself: the
paper's own Data Availability statement says *"the authors do not have
permission to share data"* and there is no code link anywhere in the paper
or its references. **This remains "PUGCN adapted to canonical IEMOCAP
forecasting," never a reproduction.**

## What changed from v1 (full diff, so nothing is quietly lost)

| Component | v1 (wrong — abstract-only guess) | v2 (rebuilt from the real paper) |
|---|---|---|
| Base model | Generic frozen sentence encoder (roberta-large) | **BART** (`facebook/bart-base`), used in the paper's own two modes |
| History encoder | Per-speaker GRU bank + global GRU (invented) | **Self-attention Transformer** over utterance representations (Eq. 6-9) |
| Interlocutor info | Edge-typing in a GRU graph | **Speaker identity token prepended to each utterance's own text** before encoding (Eq. 2) |
| Pseudo-utterance task | Masked-**history** self-supervision (avoided real target text entirely) | **BART-Generation autoregressively generates real words**, trained via teacher-forced cross-entropy against the **real target utterance's actual text** (Eq. 13-15) — now confirmed as what the paper actually does |
| Contrastive loss | Generic SupCon | The paper's specific **detach-and-duplicate trick** (Eq. 10) that doubles N→2N so every sample has a guaranteed positive |
| Loss weights | Guessed (0.2/0.3) | **Paper's actual values**: α=0.2, β=0.4 (MELD/EmoryNLP config, closest available since paper doesn't cover IEMOCAP) |
| Learning rate | Guessed (3e-4) | **Paper's actual value**: 1e-5 |

## The design decision that almost went wrong (read this)

The paper's PUG loss (Eq. 15) genuinely needs the real target utterance's
text as training supervision — that's confirmed now, not speculated. The
question is: **does that text ever influence what the classifier sees?**

**It must not, and in this implementation it structurally cannot** —
verified by `smoke_test.py`'s `test_leakage_separation_by_direct_value_comparison`,
which runs `forward_sample()` twice on the identical sample, once with
`real_target_text=None` and once with a real string, and asserts every
tensor sent to `classify_encode()` (the only thing that reaches the
classifier) is **bit-for-bit identical** between the two runs — not "looks
the same," literally `torch.equal()`. The final logits are identical too.

**How this works:** the classification path always uses `h_pseudo`, built
from `generate_pseudo_ids()` — free-running `.generate()`, which only ever
sees the history + speaker prompt, never real target text (there's no
parameter for it to receive). `pug_loss` is a completely separate quantity,
computed from a separate teacher-forced forward pass, that only trains the
generator's own word-prediction head. It's summed into the total loss for
backprop, but it never touches `h_pseudo` or the classifier's input.

**I nearly built this wrong.** My first draft of this design (in this same
session, before writing any code) considered reusing the teacher-forced
decoder's hidden states as `h_pseudo` directly — it would have been faster
(no second `.generate()` call) and seemed like a reasonable shortcut. It
would also have been a real, silent leakage bug: those hidden states are
computed by a decoder that literally had the real target tokens as its
input at every position. I caught this before writing it by re-reading the
paper's own framing — the pseudo-utterance is described as *replacing* the
missing utterance "throughout the forecasting process," implying the
classifier only ever sees the generated stand-in, at train and eval alike,
not real text during training. State this reasoning explicitly if asked
why this is slower than it could be — it's a deliberate safety choice, not
an oversight.

## Architecture (matches paper Section 3, Fig. 2)

1. **Interlocutor identity-aware encoding** (`classify_encode`): each
   utterance is rendered as `"{speaker}: {text}"`, tokenized, fed into BOTH
   BART's encoder and decoder, then max-pooled over the decoder's output
   sequence (Eq. 2-5).
2. **Conversation-level Transformer** (`ConversationEncoder`): self-attention
   (Eq. 6-9) over `[h_1, ..., h_{i-1}, h_pseudo]` — real history
   representations plus the pseudo-utterance's representation standing in
   for the target. The last position's output is the classification input.
3. **Emotion-level contrastive learning** (`supcon_loss_doubled`): Eq. 10-12
   exactly, including the detach-and-duplicate batch-doubling trick.
4. **Pseudo-utterance generation** (`pug_loss` + `generate_pseudo_ids`):
   Eq. 13-15, split into the two structurally-separate paths described
   above.

## Deviations from the paper (complete list)

1. **BART size unspecified in the paper** (never states base vs. large in
   the visible text) — `facebook/bart-base` chosen for tractability, not
   verified against the authors' actual choice.
2. **Dataset mismatch** — paper benchmarks MELD/DailyDialog/EmoryNLP; this
   adapts to IEMOCAP (canonical 100/20/31 split), never used in the paper.
   α/β weights are borrowed from the paper's MELD/EmoryNLP config as the
   closest available match, not derived for IEMOCAP.
3. **Contrastive temperature (τ) not numerically specified** in the paper
   ("a scalar temperature parameter") — defaulted to 0.1.
4. **Epoch count, dropout, per-utterance/history token-length caps** — not
   stated in the paper (which reports "optimal checkpoint from a validation
   set, averaged over 5 seeds," not a fixed epoch count) — this folder's
   `config.yaml` values are reasonable defaults, explicitly flagged as
   guesses in the file's own comments.
5. **`load_train_target_texts` only supports format-A pkls** — extend it
   (see its docstring) if your IEMOCAP pkl uses a different raw format.
6. **No padded-tensor batching across samples** — same structural reason as
   P2/P4-v1 (variable-length histories); `batch_size` in config is realized
   as gradient accumulation. Within a sample, history utterances ARE
   batched together in one BART call (see `forward_sample`), which v1
   didn't have a direct equivalent of.
7. **Single-beam greedy generation** (`num_beams=1`) for pseudo-utterances,
   for compute tractability — the paper doesn't specify beam search
   settings either way.

## Config (`config.yaml`)

Values taken directly from the paper are labeled in the file's own
comments; everything else is flagged as this implementation's own choice.

## Notebooks

`colab.ipynb` (T4 16GB, `bart-base`, heavily subsampled train/dev via
`--max_train_samples 300 --max_dev_samples 60` — a smoke/sanity run, not a
paper-quality result, see the notebook's own warning cell) and
`nautilus.ipynb` (full config, full data — budget hours, see Compute
estimate below). **Test is never subsampled by either notebook.**

## Usage

```bash
pip install -r requirements.txt

python run.py --mode train --data_path /path/to/iemocap.pkl --config config.yaml
python run.py --mode eval  --data_path /path/to/iemocap.pkl --config config.yaml \
    --model_path outputs/best.pt --save_path outputs/predictions.json

cd ../p0_evaluation && python run_eval.py --preds ../p4_pugcn/outputs/predictions.json
```

## Smoke test

```bash
python smoke_test.py
```

Five checks — no real BART download needed (no network access in this
sandbox), using a mock BART-like object with a real, already-unit-tested
`ConversationEncoder` plugged in:
1. `supcon_loss_doubled` — finite/positive/differentiable even for a
   singleton-in-batch class (the paper's stated motivation for the
   doubling trick), plus a same-vs-distinct-label loss-ordering sanity
   check (caught and corrected my own wrong prediction of the direction
   during development — see inline comments for the actual reasoning).
2. `ConversationEncoder` — shape and gradient-flow check.
3. `load_train_target_texts` — proven to return ONLY train-split text
   against a synthetic pkl containing a planted `"SECRET_TEST_TEXT"` string
   in the dev/test dialogues, which must never appear in the output.
4. **The leakage-separation proof** — the most important test in this
   folder, described above.
5. Source-level check that `--mode eval` never calls
   `load_train_target_texts()`.

Before a real run: the BART download/tokenization/generation calls
themselves are standard HF usage and weren't exercised here (no network) —
low risk, but confirm `facebook/bart-base` loads and `.generate()` behaves
as expected in your actual environment first.

## Compute estimate (materially higher than P2/P3/P4-v1)

Unlike P2 (frozen features) or v1's design, **BART is fine-tuned
end-to-end here, and every training step runs a `.generate()` call** (for
`h_pseudo`) in addition to a teacher-forced forward pass (for `pug_loss`)
and the classification-mode encoding of every history utterance. This is
the heaviest of the five baselines by a real margin.

| Stage | Compute | Est. time |
| --- | --- | --- |
| Training | Full BART fine-tuning + per-step autoregressive generation, single ordinary GPU (12GB+ recommended for bart-base) | Several hours to ~1 day on IEMOCAP-scale data, depending on `epochs` and history lengths — budget more than any other baseline in this project |
| Eval | Same per-sample cost as one training step's forward (generation + classification), no backward pass | Tens of minutes for the 1592-point test set |

If this is too slow in practice, the cheapest lever is reducing
`gen_max_new_tokens` and/or `max_history_chars` in `config.yaml` before
touching the architecture itself.


## Corrected implementation notes
- Supports the standard DialogueRNN tuple-format `IEMOCAP.pkl` and verifies full train-target-text alignment.
- Target padding tokens are masked to `-100` in the PUG loss.
- Generation-history truncation is token-based (`max_history_tokens`), not character-based.
- Uses one conversation-level Transformer layer, matching the paper description.
- This remains a **paper-based IEMOCAP adaptation**, not an exact reproduction: the paper does not provide an official implementation or all IEMOCAP-specific hyperparameters, and its separate full-dialogue ERC pretraining stage is not reproduced here. Report this deviation explicitly.

## Compute
Recommended: one A100 80 GB. An A100 40 GB may require batch size 1 and gradient accumulation. Because free-running BART generation occurs during training and development evaluation, expect approximately 12-36+ hours for a full run.

## Final result status

No finalized canonical test result is included in this package. Result cells are intentionally left blank until the H100 run and canonical evaluation are complete.
