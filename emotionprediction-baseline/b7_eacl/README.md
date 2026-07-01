# B7 - EACL (Emotion-Anchored Contrastive Learning)

**Paper:** Emotion-Anchored Contrastive Learning Framework for Emotion Recognition in Conversation. Yu, Guo, Wu, Dai.

**Paper link:** https://arxiv.org/abs/2403.20289

**Original code:** https://github.com/Yu-Fangxu/EACL

**Year / Conference:** 2024 (Findings of NAACL 2024, pp. 4521-4534)

**Original task:** ERC: a SimCSE-RoBERTa encoder with learnable per-emotion anchor vectors, trained with a supervised contrastive loss plus an anchor-separation loss; classify each utterance by nearest emotion anchor.

**What I did:** Reimplemented following the paper and adapted to forecasting by changing only the data (read the history, predict the next emotion). Best checkpoint selected on dev, test evaluated once at the end (no test leakage in model selection).

**Settings:** sup-simcse-roberta-large encoder, 8 epochs, seed 42. Best-dev checkpoint selection.

## Results (prior weighted F1, t>=1, no x_t)

| Dataset | Weighted F1 | Macro F1 | Accuracy | Parse-fail |
| --- | --- | --- | --- | --- |
| IEMOCAP | 0.5921 | 0.6007 | 0.6024 | 0.0% |

Note: dev score peaked at epoch 1 and did not improve after, so the contrastive training added little on this task; it leans on the pretrained encoder.
