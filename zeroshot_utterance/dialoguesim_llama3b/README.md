# Dialogue-sim Inference - Llama-3.2-3B (repo-exact model)

The dialogue-sim prompting recipe with their eval script's exact default model (meta-llama/Llama-3.2-3B-Instruct). Same prompts and sampling as the Qwen run.

Note: Llama is gated on HuggingFace - accept the license and set HF_TOKEN before running.

```bash
export HF_TOKEN=hf_...
python dialoguesim_nextutt.py --pkl IEMOCAP_features.pkl --model meta-llama/Llama-3.2-3B-Instruct \
  --labels_in_history 1 --out results.json --preds_out preds.jsonl
```

Tag format compliance: only 4.1% of generations closed the <dialogue> tag (91.7% left it unclosed, 4.1% never emitted it); predictions were recovered by fallback extraction. The 3B holds the format far worse than the 7B (100% clean) - a model-size observation worth noting.
Postprocessing: leading speaker-tag prefixes stripped from 28.8% of predictions (same uniform rule as the Qwen run). The _stripped2 result files are the reported numbers.
