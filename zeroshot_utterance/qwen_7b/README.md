# Zero-shot Qwen2.5-7B

Local HF model generates the next utterance from the causal history. Greedy decoding, 64 max new tokens, history window 12, past gold emotion tags shown in the prompt (labels_in_history=1).

Run (GPU):
```bash
pip install "transformers>=4.40,<4.46" accelerate sentencepiece sacrebleu rouge_score nltk bert_score sentence-transformers
python generate_nextutt.py --pkl IEMOCAP_features.pkl --model Qwen/Qwen2.5-7B-Instruct \
  --labels_in_history 1 --out results.json --preds_out preds.jsonl
```
transformers must be <4.46 (newer versions broke Qwen batched generation and changed BERTScore values).

Result files: `nextutt_zeroshot_results.json`, `nextutt_zeroshot_preds.jsonl`
