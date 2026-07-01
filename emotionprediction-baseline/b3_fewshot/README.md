# B3 - Few-shot LLM (k=6)

**Paper:** No paper (standard in-context few-shot baseline).

**Paper link:** -

**Original code:** -

**Year / Conference:** -

**Original task:** Same as B2 plus a fixed set of in-context examples drawn from train only.

**What I did:** Added 6 label-stratified examples (fixed, drawn from the training set only, no per-query retrieval, so no leakage) to the prompt, then predicted the next emotion. Inference only.

**Settings:** Inference only, no training. k=6 examples from train, greedy, up to 64 new tokens.

## Results (IEMOCAP, prior weighted F1, t>=1, no x_t)

| Model | Weighted F1 | Macro F1 | Accuracy | Parse-fail |
| --- | --- | --- | --- | --- |
| Llama-3.1-8B | 0.5739 | 0.5742 | 0.5829 | 0.4% |
| GPT-4o-mini | 0.5766 | 0.5777 | 0.5930 | 0.0% |
| Qwen2.5-3B | 0.6223 | 0.6183 | 0.6288 | 1.1% |
