# Utterance prediction experiments

This directory organizes three related but distinct evaluations:

1. `zeroshot_utterance/`: generate the next utterance and compare it with the gold next utterance using similarity metrics. This is the existing folder moved unchanged from the repository root.
2. `emotionpredictionfromutterance/`: generate one utterance, emotion-label that utterance alone, and compare with the gold emotion.
3. `multipleutteranceprediction/`: keep history and reason over one or four simulated future utterances before predicting emotion.
