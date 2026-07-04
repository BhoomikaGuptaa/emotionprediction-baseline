# B6 - InstructERC

**Paper:** InstructERC: Reforming Emotion Recognition in Conversation with Multi-task Retrieval-Augmented Large Language Models. Lei, Dong, Wang, Wang, Wang.

**Paper link:** https://arxiv.org/abs/2309.11911

**Original code:** https://github.com/LIN-SHANG/InstructERC

**Year / Conference:** 2023 (arXiv; Sep 2023)

**Original task:** Emotion recognition in conversation (ERC): given an utterance, classify its emotion. An LLM with LoRA fine-tuning plus a retrieval template that adds similar labelled training examples to the prompt.

**What I did:** Reimplemented following the paper and adapted to forecasting: query = labelled causal history + named next speaker, target = the next emotion. Retrieval pulls demos from the training set only, excluding any demo from the query's own conversation (leakage-checked). Retrained (LoRA SFT), not inference.

**Settings:** Qwen2.5-3B-Instruct + LoRA, k=4 retrieval, 3 epochs, seed 42, max_seq_length 2048. IEMOCAP split 100/20/31 conv; MELD 1038/114/280 conv; EmoryNLP 713/99/85 scenes (note: larger than some prior splits).

## Results (prior weighted F1, t>=1, no x_t)

| Dataset | Weighted F1 | Macro F1 | Accuracy | Parse-fail |
| --- | --- | --- | --- | --- |
| IEMOCAP | 0.6702| 0.6623| 0.6677| 0.9%|
| MELD | 0.4107|0.2559 |0.4236 |0.04% |
| EmoryNLP | 0.2646| 0.2365|0.2727 | 0.0%|
