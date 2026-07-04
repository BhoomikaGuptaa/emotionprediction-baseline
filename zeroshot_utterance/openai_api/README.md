# Zero-shot via OpenAI API

Same task, same prompts, same 1592 points, same metrics - the generator is an OpenAI chat model (default gpt-4o-mini, temperature 0).

Cost for the full 1592-point run: gpt-4o-mini ~$0.15-0.25, gpt-4o ~$2-3.

Run:
```bash
export OPENAI_API_KEY=sk-...
pip install openai sacrebleu rouge_score nltk bert_score sentence-transformers
python openai_nextutt.py --pkl IEMOCAP_features.pkl --model gpt-4o-mini \
  --labels_in_history 1 --out results.json --preds_out preds.jsonl
```

Result files: (added after the run)
