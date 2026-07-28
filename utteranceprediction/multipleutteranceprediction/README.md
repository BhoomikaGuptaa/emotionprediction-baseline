# Multiple-utterance prediction for emotion forecasting

This section keeps the observed dialogue history available while using simulated next utterances as additional prospective evidence.

- `original_four_candidate_reasoning/`: completed earlier four-candidate run with corrected parser.
- `matched_c1_vs_c4/`: clean controlled experiment comparing one versus four candidates from the same generation set.

The meaningful comparisons are:

- History-only A versus C1: whether one simulated future adds information beyond history.
- C1 versus C4: whether four possible futures help more than one.
- History-only A versus C4: whether multi-future reasoning improves direct forecasting.
