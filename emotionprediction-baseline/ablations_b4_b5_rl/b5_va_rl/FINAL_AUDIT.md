# Final audit status

B5 is the continuous direct-GRPO counterpart to B4:

- 0.2 strict format
- 0.2 valid label
- 0.6 training-derived V/A similarity

The only intended reward difference from B4 is the task term:

- B4: exact categorical match
- B5: continuous V/A proximity

The V/A centroids are computed from original IEMOCAP dimensional annotations restricted to canonical training dialogues only. B5 requires either `--emotion_eval_dir` or a precomputed train-only `--va_centroids_json`.

Verified safeguards:

- completion-only SFT with prompt labels masked to -100;
- strict `<emotion>...</emotion>` parsing;
- strict target/completion alignment;
- generic metadata expansion for sample IDs;
- unique per-target sample IDs;
- development-selected GRPO checkpoint;
- final test evaluation after checkpoint selection;
- saved centroids, counts, similarity matrix, source metadata and data hash.

Run `python run.py --mode self_test` and then a 4-step real-data smoke test before the full job.
