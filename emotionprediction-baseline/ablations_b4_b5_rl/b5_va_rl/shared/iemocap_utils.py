"""
shared/iemocap_utils.py  (causal-aligned, v4)
=============================================
Canonical loader + label handling + evaluation for baselines B1-B5.

Task: CAUSAL next-emotion prediction (matches the DPM paper's problem setting).
  Predict p(y_t | H_t, i_t) WITHOUT accessing x_t.
  H_t = {(x_s, i_s, y_s) : s < t}  -> history carries text, speaker, AND gold emotion.
  i_t = identity of the next speaker (known; turn-taking is observed).
  Scored on all turns with t >= 1 (prior weighted F1).

What changed vs. the earlier version (and why):
  1. Sample now carries history_emotions (gold y_s for s<t) and target_speaker (i_t).
     The earlier prompts dropped prior emotions; the DPM ablation shows the most
     recent observed emotion is the single strongest predictor (-0.1436 F1 when
     removed). Omitting it both mismatches H_t and rigs the comparison against us.
  2. format_history() renders "Speaker X (emotion): text" so the LLM sees H_t.
  3. Split is 100 train / 20 val / 31 test CONVERSATIONS, carved at the dialogue
     level from the standard pkl, matching DPM Appendix C. No utterance-level dev
     slice (that leaked one dialogue across train/dev in the old B1-B3 loader).
  4. evaluate() is parse-failure aware: an unparseable generation is scored WRONG
     (sentinel id = NUM_LABELS) instead of being silently mapped to neutral, and
     the parse-failure rate is reported. This stops the metric being gamed by the
     fallback on a dataset where neutral is common.
  5. Numeric label order is the canonical DialogueRNN order. ALWAYS confirm against
     inspect_pkl() output for your specific pkl before trusting any number.
"""

import re
import math
import json
import pickle
import inspect
from dataclasses import dataclass, fields as _dc_fields
from collections import Counter


# ---- TRL / transformers version-compat helpers (used by B4/B5) --------------
_CONFIG_ALIASES = {
    "max_seq_length": ("max_seq_length", "max_length"),
    "max_length":     ("max_length", "max_seq_length"),
}


def _accepted_names(cls):
    names = set()
    try:
        names |= {f.name for f in _dc_fields(cls)}
    except TypeError:
        pass
    try:
        names |= set(inspect.signature(cls.__init__).parameters.keys())
    except (TypeError, ValueError):
        pass
    return names


def build_config(cls, **kwargs):
    accepted = _accepted_names(cls)
    out = {}
    for k, v in kwargs.items():
        if k in accepted:
            out[k] = v
        elif k in _CONFIG_ALIASES:
            for alt in _CONFIG_ALIASES[k]:
                if alt in accepted:
                    out[alt] = v
                    break
    return cls(**out)


def build_trainer(cls, **kwargs):
    accepted = set(inspect.signature(cls.__init__).parameters.keys())
    if "tokenizer" in kwargs and "tokenizer" not in accepted and "processing_class" in accepted:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
    elif "processing_class" in kwargs and "processing_class" not in accepted and "tokenizer" in accepted:
        kwargs["tokenizer"] = kwargs.pop("processing_class")
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(**kwargs)


# ---- Canonical label set ----------------------------------------------------
EMOTION_LABELS = ["neutral", "frustration", "sadness", "anger", "excited", "happiness"]
LABEL2ID       = {e: i for i, e in enumerate(EMOTION_LABELS)}
ID2LABEL       = {i: e for e, i in LABEL2ID.items()}
NUM_LABELS     = len(EMOTION_LABELS)
PARSE_FAIL_ID  = NUM_LABELS          # sentinel: never equals any true label -> counts as wrong

LABEL_NORMALIZE = {
    "anger": "anger", "angry": "anger", "ang": "anger",
    "happiness": "happiness", "happy": "happiness", "hap": "happiness",
    "sadness": "sadness", "sad": "sadness",
    "frustration": "frustration", "frustrated": "frustration", "fru": "frustration",
    "excited": "excited", "excitement": "excited", "exc": "excited",
    "neutral": "neutral", "neu": "neutral",
}

# Canonical DialogueRNN IEMOCAP numeric order: hap, sad, neu, ang, exc, fru.
NUMERIC_LABEL_NORMALIZE = {
    0: "happiness", 1: "sadness", 2: "neutral",
    3: "anger",     4: "excited", 5: "frustration",
}


def normalize_label(raw):
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return NUMERIC_LABEL_NORMALIZE.get(raw)
    if isinstance(raw, float) and float(raw).is_integer():
        return NUMERIC_LABEL_NORMALIZE.get(int(raw))
    txt = str(raw).strip().lower()
    if txt.isdigit():
        return NUMERIC_LABEL_NORMALIZE.get(int(txt))
    return LABEL_NORMALIZE.get(txt, None)


# ---- Literature-grounded V/A coordinates + similarity (B5) -----------------
# Warriner, Kuperman & Brysbaert (2013), word-level human norms on a 1--9 scale.
# We use the exact English label lemmas used by this project and map each axis
# to [-1, 1] with (rating - 5) / 4. These are lexical norms, not IEMOCAP
# utterance-level centroids; that limitation is recorded in B5 metadata.
WARRINER_VA_RAW_1_TO_9 = {
    "neutral":     (5.50, 3.45),
    "frustration": (2.45, 5.94),
    "sadness":     (2.40, 2.81),
    "anger":       (2.50, 5.93),
    "excited":     (8.11, 6.43),
    "happiness":   (8.48, 6.50),
}

EMOTION_VA = {
    label: ((v - 5.0) / 4.0, (a - 5.0) / 4.0)
    for label, (v, a) in WARRINER_VA_RAW_1_TO_9.items()
}

def _build_va_sim():
    distances = {}
    for ei in EMOTION_LABELS:
        for ej in EMOTION_LABELS:
            vi, ai = EMOTION_VA[ei]
            vj, aj = EMOTION_VA[ej]
            distances[(ei, ej)] = math.dist((vi, ai), (vj, aj))
    max_d = max(distances.values())
    if max_d <= 0:
        raise RuntimeError("Degenerate V/A coordinate set")
    return {
        ei: {ej: max(0.0, 1.0 - distances[(ei, ej)] / max_d) for ej in EMOTION_LABELS}
        for ei in EMOTION_LABELS
    }

VA_SIM = _build_va_sim()

# ---- Data structure ---------------------------------------------------------
@dataclass
class Sample:
    dialogue_id:       str
    history:           list   # texts u_0 .. u_{t-1}
    history_speakers:  list   # speakers i_0 .. i_{t-1}
    history_emotions:  list   # gold labels y_0 .. y_{t-1}  (canonical strings)
    target_speaker:    str    # i_t  (known: who speaks next)
    target_emotion:    str    # y_t  (to predict)
    target_emotion_id: int


def format_history(history, speakers, emotions=None, max_turns=10):
    """Render H_t. Each prior turn shows speaker and, when available, gold emotion."""
    history  = history[-max_turns:]
    speakers = speakers[-max_turns:]
    if emotions is not None:
        emotions = emotions[-max_turns:]
    lines = []
    for i, (spk, utt) in enumerate(zip(speakers, history)):
        if emotions is not None and i < len(emotions) and emotions[i] is not None:
            lines.append(f"Speaker {spk} ({emotions[i]}): {utt}")
        else:
            lines.append(f"Speaker {spk}: {utt}")
    return "\n".join(lines)


# ---- Split helper: 100 / 20 / 31 conversations ------------------------------
def _carve_val(train_vids, n_val=20):
    """Deterministically pull n_val dialogue ids out of train for validation."""
    train_sorted = sorted(train_vids)
    if len(train_sorted) <= n_val:
        n_val = max(1, int(0.1 * len(train_sorted))) if len(train_sorted) > 10 else 0
    val = set(train_sorted[len(train_sorted) - n_val:]) if n_val else set()
    train = set(train_sorted) - val
    return train, val


# ---- Loaders (auto-detect pkl format) ---------------------------------------
def load_iemocap_pkl(pkl_path: str) -> dict:
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    if isinstance(raw, dict) and "train" in raw:
        return _load_format_a(raw)
    if isinstance(raw, dict) and ("videoSentence" in raw or "trainVid" in raw):
        return _load_format_b(raw)
    if isinstance(raw, (list, tuple)) and len(raw) >= 9:
        return _load_format_dialoguernn_tuple(raw)
    keys = list(raw.keys()) if isinstance(raw, dict) else f"{type(raw)}"
    raise ValueError(f"Unrecognised pickle format. Top-level: {keys}. Run inspect_pkl().")


def _emit_samples(split_list, vid_id, utts, spks, emos):
    """Build causal samples for one dialogue: t in 1..len-1, target = label[t]."""
    out = []
    canon_emos = [normalize_label(e) for e in emos]
    for t in range(1, len(utts)):
        if t >= len(canon_emos):
            continue
        y_t = canon_emos[t]
        if y_t is None:
            continue
        out.append(Sample(
            dialogue_id       = f"{vid_id}_t{t}",
            history           = list(utts[:t]),
            history_speakers  = list(spks[:t]),
            history_emotions  = list(canon_emos[:t]),     # gold y_{<t}
            target_speaker    = spks[t] if t < len(spks) else "?",
            target_emotion    = y_t,
            target_emotion_id = LABEL2ID[y_t],
        ))
    return out


def _load_format_dialoguernn_tuple(raw):
    video_speakers = raw[1]
    video_labels   = raw[2]
    video_sentence = raw[6]
    train_vids     = list(raw[7])
    test_vids      = set(raw[8])

    train_set, val_set = _carve_val(train_vids, n_val=20)
    splits = {"train": [], "dev": [], "test": []}
    for vid, utts in video_sentence.items():
        emos = video_labels.get(vid, [])
        spks = video_speakers.get(vid, ["M" if i % 2 == 0 else "F" for i in range(len(utts))])
        if vid in test_vids:      split = "test"
        elif vid in val_set:      split = "dev"
        elif vid in train_set:    split = "train"
        else:                     continue          # unlisted -> drop, no leakage
        splits[split].extend(_emit_samples(splits[split], vid, utts, spks, emos))
    return splits


def _load_format_a(raw):
    # Dict already split into train/dev/test dialogue lists.
    splits = {}
    for split in ["train", "dev", "test"]:
        samples = []
        for dlg in raw.get(split, []):
            utts = dlg.get("utterance", dlg.get("utterances", []))
            emos = dlg.get("emotion",   dlg.get("emotions",   []))
            spks = dlg.get("speaker",   dlg.get("speakers",
                   ["M" if i % 2 == 0 else "F" for i in range(len(utts))]))
            cid  = dlg.get("conv_id", dlg.get("vid", "unknown"))
            samples.extend(_emit_samples(samples, cid, utts, spks, emos))
        splits[split] = samples
    return splits


def _load_format_b(raw):
    sents   = raw.get("videoSentence", {})
    labels  = raw.get("videoLabels",   {})
    spkrs   = raw.get("videoSpeakers", {})
    train_vids = list(raw.get("trainVid", []))
    test_vids  = set(raw.get("testVid",  []))
    train_set, val_set = _carve_val(train_vids, n_val=20)

    splits = {"train": [], "dev": [], "test": []}
    for vid, utts in sents.items():
        emos = labels.get(vid, [])
        spks = spkrs.get(vid, ["M" if i % 2 == 0 else "F" for i in range(len(utts))])
        if vid in test_vids:      split = "test"
        elif vid in val_set:      split = "dev"
        elif vid in train_set:    split = "train"
        else:                     continue
        splits[split].extend(_emit_samples(splits[split], vid, utts, spks, emos))
    return splits


def load_iemocap_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    splits = {"train": [], "dev": [], "test": []}
    for entry in data:
        split  = entry.get("split", "train")
        dlg_id = entry["dialogue_id"]
        utts   = [u["text"] for u in entry["utterances"]]
        spks   = [u.get("speaker", "M" if i % 2 == 0 else "F")
                  for i, u in enumerate(entry["utterances"])]
        emos   = [u.get("emotion") for u in entry["utterances"]]
        splits[split].extend(_emit_samples(splits[split], dlg_id, utts, spks, emos))
    return splits


# ---- Inspect ----------------------------------------------------------------
def inspect_pkl(pkl_path: str):
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    print(f"Type: {type(raw)}")
    if isinstance(raw, (list, tuple)):
        print(f"Length: {len(raw)}")
        for i, v in enumerate(raw):
            line = f"  raw[{i}] -> {type(v).__name__}"
            if hasattr(v, "__len__"):
                line += f", len={len(v)}"
            if isinstance(v, dict) and v:
                k = next(iter(v))
                line += f"; sample key={k}; val={str(v[k])[:80]}"
            print(line)
    elif isinstance(raw, dict):
        print(f"Keys: {list(raw.keys())}")
    print("\nReminder: confirm the numeric label order matches "
          "NUMERIC_LABEL_NORMALIZE (0=hap,1=sad,2=neu,3=ang,4=exc,5=fru).")


# ---- Emotion-shift (ES) vs no-shift (woES) breakdown -------------------------
# Standard ERC reporting convention (BERT-ERC, AAAI 2023; DAG-ERC): a "shift"
# is defined relative to the SAME speaker's own previous utterance, not the
# literal previous turn in the transcript. On IEMOCAP's dyadic structure the
# immediately preceding turn is often the OTHER speaker, so grabbing
# history_emotions[-1] naively would frequently measure "does my prediction
# differ from a stranger's emotion" instead of genuine within-speaker shift.
# This mirrors the exact same-speaker lookback already used by the persistence
# floor (b1_floors/code_persistence/run.py) -- that baseline computes the
# thing this function names, so both stay consistent by construction.

def prev_speaker_emotion(sample):
    """Most recent gold emotion from the SAME speaker who takes turn t.
    STRICT same-speaker only -- NO any-speaker fallback. Returns None if
    the target speaker has no prior turn yet in this dialogue (their first
    appearance). A prior version of this function fell back to the most
    recent emotion from ANY speaker in that case, which silently
    reintroduced exactly the cross-speaker conflation problem the
    same-speaker shift definition exists to avoid in the first place --
    comparing the target speaker's first-ever emotion against a different
    person's emotion and calling the result "shift" or "no-shift" is not a
    meaningful same-speaker comparison. Fixed: no fallback, full stop."""
    spk = sample.target_speaker
    for i in range(len(sample.history) - 1, -1, -1):
        if sample.history_speakers[i] == spk and sample.history_emotions[i] is not None:
            return sample.history_emotions[i]
    return None


def is_emotion_shift(sample):
    """True/False = shift/no-shift vs. the same-speaker prior emotion.
    None = undefined -- either no labelled turn exists at all, or (just as
    importantly) the target speaker has no prior turn yet in this dialogue.
    Both cases are structurally identical here: prev_speaker_emotion
    returns None for either, and shift status is genuinely undefined, not
    guessable from a different speaker's history. Excluded from both ES
    and woES buckets, never silently forced into one."""
    pe = prev_speaker_emotion(sample)
    if pe is None:
        return None
    return pe != sample.target_emotion


def compute_shift_breakdown(samples, y_pred, model_name="Model", verbose=True):
    """Weighted/macro F1 + accuracy split into ES (emotion-shift) vs. woES
    (no-shift) subsets, per standard ERC reporting convention.
    `samples` and `y_pred` must be aligned: samples[i] <-> y_pred[i].
    Returns {"es": {...}, "woes": {...}, "undefined_n": int, "shift_flags": [...]}."""
    from sklearn.metrics import f1_score, accuracy_score

    assert len(samples) == len(y_pred), "samples and y_pred must be the same length and order"
    labels = list(range(NUM_LABELS))

    shift_flags = [is_emotion_shift(s) for s in samples]
    es_idx   = [i for i, f in enumerate(shift_flags) if f is True]
    woes_idx = [i for i, f in enumerate(shift_flags) if f is False]
    undefined_n = sum(1 for f in shift_flags if f is None)

    def _bucket(idx):
        if not idx:
            return {"weighted_f1": None, "macro_f1": None, "accuracy": None, "n": 0}
        yt = [samples[i].target_emotion_id for i in idx]
        yp = [y_pred[i] for i in idx]
        return {
            "weighted_f1": round(float(f1_score(yt, yp, average="weighted", labels=labels, zero_division=0)), 4),
            "macro_f1":    round(float(f1_score(yt, yp, average="macro",    labels=labels, zero_division=0)), 4),
            "accuracy":    round(float(accuracy_score(yt, yp)), 4),
            "n": len(idx),
        }

    es_stats, woes_stats = _bucket(es_idx), _bucket(woes_idx)

    if verbose:
        print(f"\n{'-'*55}\n  {model_name} -- ES / woES breakdown (same-speaker shift def.)\n{'-'*55}")
        print(f"  Emotion-shift (ES)   n={es_stats['n']:<5} weighted F1={es_stats['weighted_f1']}")
        print(f"  No-shift    (woES)   n={woes_stats['n']:<5} weighted F1={woes_stats['weighted_f1']}")
        if undefined_n:
            print(f"  Undefined (no prior labelled turn) excluded: {undefined_n}")

    return {"es": es_stats, "woes": woes_stats, "undefined_n": undefined_n,
            "shift_flags": ["shift" if f is True else "noshift" if f is False else "undefined"
                             for f in shift_flags]}


# ---- Evaluation -------------------------------------------------------------
def evaluate(y_true, y_pred, model_name="Model", raw_outputs=None, save_path=None,
             samples=None):
    """Prior weighted F1 over t>=1, plus macro/acc/per-class and parse-fail rate.
    Parse failures must arrive as PARSE_FAIL_ID in y_pred (counted as wrong).

    Pass `samples` (the same Sample list y_true/y_pred were built from, same
    order) to additionally compute and save the ES/woES emotion-shift
    breakdown -- the paper's primary thesis metric. Backward compatible:
    omitting `samples` reproduces the exact prior behavior/output."""
    from sklearn.metrics import f1_score, accuracy_score, classification_report

    n_fail = sum(1 for p in y_pred if p == PARSE_FAIL_ID)
    labels = list(range(NUM_LABELS))

    wf1 = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
    mf1 = f1_score(y_true, y_pred, average="macro",    labels=labels, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    pcf = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

    print(f"\n{'='*55}\n  {model_name}\n{'='*55}")
    print(f"  Prior Weighted F1 : {wf1:.4f}   <- PRIMARY (t>=1, no x_t)")
    print(f"  Macro F1          : {mf1:.4f}")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Parse failures    : {n_fail}/{len(y_pred)} "
          f"({100*n_fail/max(1,len(y_pred)):.1f}%, scored wrong)")
    print(f"\n  Per-class F1:")
    for i, lbl in enumerate(EMOTION_LABELS):
        print(f"    {lbl:<15} {pcf[i]:.4f}  {'#'*int(pcf[i]*20)}")
    print("\n" + classification_report(
        y_true, y_pred, labels=labels,
        target_names=EMOTION_LABELS, zero_division=0))

    results = {
        "model": model_name,
        "weighted_f1": round(float(wf1), 4),
        "macro_f1":    round(float(mf1), 4),
        "accuracy":    round(float(acc), 4),
        "parse_fail_rate": round(n_fail / max(1, len(y_pred)), 4),
        "per_class_f1": {EMOTION_LABELS[i]: round(float(pcf[i]), 4) for i in range(NUM_LABELS)},
    }

    shift_result = None
    if samples is not None:
        shift_result = compute_shift_breakdown(samples, y_pred, model_name)
        results["shift_breakdown"] = {"es": shift_result["es"], "woes": shift_result["woes"],
                                       "undefined_n": shift_result["undefined_n"]}

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        def name(idx):
            return "PARSE_FAIL" if idx == PARSE_FAIL_ID else ID2LABEL[idx]
        preds_out = []
        for i, (t, p) in enumerate(zip(y_true, y_pred)):
            row = {"true": ID2LABEL[t], "pred": name(p),
                   "raw": raw_outputs[i] if raw_outputs else None}
            if samples is not None:
                s = samples[i]
                row["dialogue_id"] = s.dialogue_id
                row["target_speaker"] = s.target_speaker
                row["shift"] = shift_result["shift_flags"][i]
            preds_out.append(row)
        with open(save_path, "w") as f:
            json.dump({"results": results, "predictions": preds_out}, f, indent=2)
        print(f"  Saved -> {save_path}")
    return results


# Backwards-compatible alias (the old B1-B3 notebook calls this name).
evaluate_predictions = evaluate


# ---- Shared prompt + parsing (identical across B2,B3,B4,B5) ------------------
SYSTEM_PROMPT = (
    "You are an expert in conversational emotion analysis.\n"
    "You are given a dialogue history. Each prior turn is labelled with the "
    "speaker and the emotion they expressed.\n"
    "Predict the emotion of the NEXT turn, spoken by the named next speaker.\n"
    "You cannot see the next utterance, only the history and who speaks next.\n\n"
    f"Valid emotions: {', '.join(EMOTION_LABELS)}\n\n"
    "Reply in this EXACT format:\n"
    "<think>\nbrief reasoning about the emotional trajectory\n</think>\n"
    "<emotion>\none_emotion_label\n</emotion>"
)


def make_prompt(history, speakers, emotions, target_speaker, max_turns=10):
    """History carries gold prior emotions; prompt names the next speaker i_t."""
    hist = format_history(history, speakers, emotions, max_turns=max_turns)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Dialogue history:\n{hist}\n\n"
        f"The next turn is spoken by Speaker {target_speaker}.\n"
        f"What emotion will Speaker {target_speaker} express?"
    )


_EMO_RE = re.compile(r"<emotion>\s*([A-Za-z]+)\s*</emotion>", re.IGNORECASE)
# Fallback scans for any known surface form, longest first so "sadness" wins
# over "sad". Includes variants (sad, happy, angry, ...) so small instruct models
# that answer with the short word are parsed correctly rather than scored wrong.
_WRD_VARIANTS = sorted(LABEL_NORMALIZE.keys(), key=len, reverse=True)
_WRD_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in _WRD_VARIANTS) + r")\b",
                     re.IGNORECASE)
_FMT_RE = re.compile(r"<think>.*?</think>.*?<emotion>.*?</emotion>", re.DOTALL)


def parse_emotion(text):
    """Return a canonical label string, or None if nothing parseable is found."""
    m = _EMO_RE.search(text)
    if m:
        lbl = normalize_label(m.group(1))
        if lbl is not None:
            return lbl
    m = _WRD_RE.search(text)
    if m:
        return normalize_label(m.group(1))
    return None


def check_format(text):
    return bool(_FMT_RE.search(text))


def pred_to_id(emotion):
    """Map a parsed emotion (or None) to an id; None -> PARSE_FAIL_ID (scored wrong)."""
    if emotion is None:
        return PARSE_FAIL_ID
    return LABEL2ID.get(emotion, PARSE_FAIL_ID)
