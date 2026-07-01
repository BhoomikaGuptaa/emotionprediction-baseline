# B2 - Zero-shot LLM

**Paper:** No paper (standard zero-shot prompting baseline).

**Paper link:** -

**Original code:** -

**Year / Conference:** -

**Original task:** Prompt an instruction-tuned LLM to name the next emotion, no examples, no training.

**What I did:** Gave the model the labelled causal history and the next speaker, asked for the next emotion. Inference only, greedy decoding.

**Settings:** Inference only, no training. Greedy, up to 64 new tokens.

## Results (IEMOCAP, prior weighted F1, t>=1, no x_t)

| Model | Weighted F1 | Macro F1 | Accuracy | Parse-fail |
| --- | --- | --- | --- | --- |
| Llama-3.2-1B | 0.4150 | 0.4051 | 0.4504 | 2.0% |
| Llama-3.1-8B | 0.5981 | 0.5894 | 0.6043 | 0.8% |
| Qwen2.5-3B | 0.5542 | 0.5541 | 0.5779 | 0.8% |
