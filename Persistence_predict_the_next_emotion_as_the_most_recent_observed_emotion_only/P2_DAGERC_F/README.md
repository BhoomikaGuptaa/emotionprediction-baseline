# P2 — DAG-ERC-F (causal DAG-ERC adaptation)

Independent folder — no dependency on any other `p*` folder beyond the
shared prediction-row format that P0 can score (see P0's README).

**Paper:** Shen, Wu, Yang, Quan, "Directed Acyclic Graph Network for
Conversational Emotion Recognition," ACL-IJCNLP 2021.
**Link:** https://arxiv.org/abs/2105.12907
**Original code:** https://github.com/shenwzh3/DAG-ERC

**Call this "DAG-ERC-F" or "DAG-ERC causal adaptation" in any writeup —
never "DAG-ERC" unqualified.** This is not a reproduction of the original
paper's numbers or its exact architecture; see Deviations below.

## What it does

Encodes each conversation's HISTORY ONLY as a directed acyclic graph: each
history utterance connects to a local window of preceding utterances
(same-speaker / different-speaker edge typing), propagated through stacked
attention-weighted GRU updates. A virtual **target node** (no real text —
a learned "unknown-utterance" vector stands in) is appended, connected to
the same local window with the target speaker's identity used for edge
typing, and its final state is classified. This is the same "can't see
x_t, but can see who speaks next" adaptation pattern as this project's
other baselines (B6/B7 in `emotionprediction-baseline`).

## Strict no-leakage guarantee

- The graph never contains the real target utterance as a node.
- `build_feature_cache()` only ever reads `Sample.history` — `Sample`
  objects structurally don't carry the target utterance's text at all
  (see `shared/iemocap_utils.py`'s `Sample` dataclass), so there's nothing
  to accidentally leak even with a coding mistake in this file.
- Only the target *speaker identity* is used for the virtual node, matching
  the task definition (next speaker is known, next content is not).
- No turn beyond t is ever read (not just "the very next turn" — nothing at
  any future index).

## Deviations from the original paper (disclosed, not hidden)

1. **Forecasting adaptation itself.** The original DAG-ERC predicts every
   node's own emotion from its own text (ERC). This predicts an unseen
   future node's emotion from a virtual node with no text (EFC/forecasting).
   This is the single largest deviation and the whole point of this folder.
2. **Frozen features, not joint fine-tuning of the graph+encoder.** Matches
   the original paper's own design (frozen RoBERTa-Large features feed a
   separate graph net) — not a deviation from the paper, but worth stating
   since many modern reimplementations jointly fine-tune everything.
3. **No padded-tensor batching.** Each conversation graph has a different
   node count and a different edge structure per sample, so this
   dependency-light version (no `torch_geometric`, no custom padding/masking
   code) processes one sample's graph at a time in a Python loop.
   `batch_size` in `config.yaml` is realized as **gradient accumulation**
   (this many samples' losses summed before one `optimizer.step()`), not
   real padded-batch tensor ops. This is a legitimate way to hit a target
   effective batch size, but it does not get the wall-clock speedup of true
   GPU batching — see the Compute section below. Flag this if scaling to a
   larger dataset (MELD) feels slow; that's where real batching would start
   to matter.
4. **Edge typing simplified to binary same/different-speaker**, generalized
   from IEMOCAP's 2-party structure to work on any number of parties (MELD/
   EmoryNLP have >2 speakers) — the original paper's exact multi-relation
   edge scheme (if it has more than 2 relation types) was not verified
   line-by-line against the original code, only against the paper's
   textual description. Confirm against `shenwzh3/DAG-ERC`'s actual edge
   construction code before claiming exact architectural parity.
5. **No text-similarity/graph-construction tricks from the original repo**
   (if any exist — e.g. any special first/last-node handling) were verified
   beyond what the paper text describes. This implementation was built from
   the paper's textual description of the mechanism (DAG + local window +
   speaker-typed edges + GRU update), not by reading the original repo's
   source line-by-line. State this plainly in any paper writeup: "adapted
   from the paper's description," not "ported from the official code."

## Config (`config.yaml`)

Starting point matches the paper's own IEMOCAP config:
```yaml
gnn_layers: 4
learning_rate: 0.0005
batch_size: 16          # gradient-accumulation grouping, see Deviation #3
epochs: 30
dropout: 0.2
seed: 42
window: 10
hidden: 300
encoder: roberta-large
```
All values overridable via CLI flags (`--epochs`, `--learning_rate`, etc.)
without editing the file.

## Notebooks

`colab.ipynb` (T4 16GB, roberta-base, subsampled train/dev, test always
full) and `nautilus.ipynb` (paper's own roberta-large/30-epoch config, full
data) — both validated, ready to open and run after editing `DATA_PATH`.
Colab uses the new `--max_train_samples`/`--max_dev_samples` flags below to
fit a normal session; **test is never subsampled** by either notebook.

## Usage

```bash
pip install -r requirements.txt

# Stage 1: cache frozen utterance features (once per dataset; needs a GPU
# for speed but will run on CPU too, just slower)
python run.py --mode cache --data_path /path/to/iemocap.pkl --config config.yaml

# Stage 2: train the graph net (checkpoint selected on dev weighted F1)
python run.py --mode train --data_path /path/to/iemocap.pkl --config config.yaml

# Stage 3: evaluate + save predictions in the P0-compatible rich format
python run.py --mode eval --data_path /path/to/iemocap.pkl --config config.yaml \
    --model_path outputs/best.pt --save_path outputs/predictions.json

# Score it with P0:
cd ../p0_evaluation && python run_eval.py --preds ../p2_dagerc/outputs/predictions.json
```

`outputs/run_metadata.json` (written after training) records the exact
command, resolved config, best dev weighted F1, timestamp, and git commit
hash — the paper-trail record for this run.

## Compute estimate

| Stage | Compute | Est. time |
| --- | --- | --- |
| 1. Feature caching | Any single GPU 8GB+ (frozen encoder, no backward pass) | 5-15 min |
| 2. Graph training | Small graph net (hidden=300, 4 layers), gradient-accumulation "batching" (see Deviation #3) | 30-90 min GPU, a few hours CPU-only |
| 3. Eval | Single forward pass per test sample | <5 min |

Cheapest of the four new baselines to add — no LoRA, no generation, no
multi-GPU coordination. Runs on any single ordinary GPU or even CPU-only
with patience.

## Smoke test

```bash
python smoke_test.py
```

Tests the model/graph logic (not the HF encoder call, which needs network
access this environment doesn't have) on synthetic in-memory features:
correct `dialogue_id` parsing, clean forward+backward over 40 epochs with
loss dropping from 1.895→0.051, and 5/5 correct predictions on the
tiny overfit set in eval mode. This exact test caught a real in-place-
tensor-mutation autograd bug during development — see Deviations /
development history for details; the fix (accumulate node states in a
Python list + `torch.stack`, rather than writing into a cloned tensor
in-place) is already applied in `run.py`.

## Naming and scope
This is a **custom DAG-inspired causal graph baseline**, not an exact reproduction of the original DAG-ERC implementation. Report it as `DAG-ERC-F (custom causal adaptation)` or `DAG-inspired causal graph baseline`. The runner now hard-fails if the frozen-feature cache encoder differs from the active config.

## Compute
Recommended: one A100 40 GB. Expect roughly 0.5-2 hours for frozen feature caching and 1-4 hours for training/evaluation. A 24 GB GPU may work with a smaller batch size.

## Final result status

No finalized canonical test result is included in this package. Result cells are intentionally left blank until the completed run is evaluated with the shared canonical protocol.
