"""Smoke tests for P5 speaker-conditioned transition baseline."""
import json, os, pickle, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from shared.iemocap_utils import Sample, LABEL2ID, EMOTION_LABELS
from run import build_transition_matrix, predict, strict_same_speaker_prev


def test_strict_same_speaker_prev():
    s = Sample(
        dialogue_id="c_t3",
        history=["u0", "u1", "u2"],
        history_speakers=["A", "B", "A"],
        history_emotions=["neutral", "neutral", "anger"],
        target_speaker="B",
        target_emotion="sadness",
        target_emotion_id=LABEL2ID["sadness"],
    )
    assert strict_same_speaker_prev(s) == "neutral"
    s2 = Sample(
        dialogue_id="c_t3b",
        history=s.history,
        history_speakers=s.history_speakers,
        history_emotions=s.history_emotions,
        target_speaker="C",
        target_emotion="happiness",
        target_emotion_id=LABEL2ID["happiness"],
    )
    assert strict_same_speaker_prev(s2) is None
    print("[PASS] strict same-speaker lookback has no any-speaker fallback")


def test_smoothing_and_fallback():
    train = []
    for i in range(3):
        train.append(Sample(f"t{i}", ["x"], ["A"], ["neutral"], "A", "anger", LABEL2ID["anger"]))
    train.append(Sample("t3", ["x"], ["A"], ["neutral"], "A", "sadness", LABEL2ID["sadness"]))
    transition, n = build_transition_matrix(train, 1.0)
    assert n == 4
    idx = {e: i for i, e in enumerate(EMOTION_LABELS)}
    assert abs(transition["neutral"][idx["anger"]] - 0.4) < 1e-9

    tier1 = Sample("q1", ["x"], ["A"], ["neutral"], "A", "anger", LABEL2ID["anger"])
    pred, tier = predict(tier1, transition, LABEL2ID["neutral"])
    assert tier == "tier1_learned_transition" and pred == LABEL2ID["anger"]

    no_history_for_target = Sample("q2", ["x"], ["A"], ["sadness"], "B", "sadness", LABEL2ID["sadness"])
    pred, tier = predict(no_history_for_target, transition, LABEL2ID["neutral"])
    assert tier == "tier2_majority_no_same_speaker_history" and pred == LABEL2ID["neutral"]
    print("[PASS] smoothing and majority fallback are correct")


def test_end_to_end_cli():
    tmp = tempfile.mkdtemp()
    try:
        dialogue = {
            "conv_id": "conv1",
            "utterance": [f"u{i}" for i in range(6)],
            "speaker": ["A", "B", "A", "B", "A", "B"],
            "emotion": ["neutral", "neutral", "anger", "neutral", "anger", "sadness"],
        }
        pkl_path = os.path.join(tmp, "synthetic.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({"train": [dialogue], "dev": [dialogue], "test": [dialogue]}, f)
        save_path = os.path.join(tmp, "predictions.json")
        r = subprocess.run(
            [sys.executable, os.path.join(HERE, "run.py"), "--data_path", pkl_path,
             "--config", os.path.join(HERE, "config.yaml"), "--save_path", save_path],
            cwd=HERE, capture_output=True, text=True,
        )
        assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
        rows = json.load(open(save_path))["predictions"]
        assert len(rows) == 5 and all("shift" in row for row in rows)
        print("[PASS] end-to-end CLI output is P0-compatible")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_strict_same_speaker_prev()
    test_smoothing_and_fallback()
    test_end_to_end_cli()
    print("\nALL SMOKE TESTS PASSED")
