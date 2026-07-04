# Dialogue-sim Inference - Qwen2.5-7B

The dialogue-sim prompting recipe (kanishkg/dialogue-sim, their eval script's thinking mode: reason in <think> tags, predict in <dialogue> tags, temperature 0.3, top_p 0.95, 512 max new tokens) run with Qwen2.5-7B-Instruct. Same model as the plain zero-shot run, so this isolates the effect of the recipe.

Run:
```bash
pip install "transformers>=4.40,<4.46" accelerate sentencepiece sacrebleu rouge_score nltk bert_score sentence-transformers
python dialoguesim_nextutt.py --pkl IEMOCAP_features.pkl --model Qwen/Qwen2.5-7B-Instruct \
  --labels_in_history 1 --out results.json --preds_out preds.jsonl
```

Tag format compliance: 100% clean (all 1592 generations produced a closed <dialogue> tag).
Postprocessing: leading speaker-tag prefixes (e.g. "M (frustration):"), an artifact of the labelled-history prompt, were stripped from 98.7% of predictions before scoring (uniform rule across both dialogue-sim runs). Result files: the _stripped2 versions are the reported numbers.
