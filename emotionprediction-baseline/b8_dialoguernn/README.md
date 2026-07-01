# B8 - DialogueRNN

**Paper:** DialogueRNN: An Attentive RNN for Emotion Detection in Conversations. Majumder, Poria, Hazarika, Mihalcea, Gelbukh, Cambria.

**Paper link:** https://ojs.aaai.org/index.php/AAAI/article/view/4657

**Original code:** https://github.com/declare-lab/conv-emotion

**Year / Conference:** 2019 (AAAI 2019, pp. 6818-6825)

**Original task:** ERC: a recurrent model that tracks per-speaker and global states across the dialogue (bidirectional) and labels each utterance's own emotion.

**What I did:** Reimplemented following the paper and adapted to forecasting: shifted labels to predict the next turn, masked the last turn, used the forward-only version so it cannot read the future turn.

**Settings:** Forward-only DialogueRNN, 30 epochs/seed, seeds 42/43/44 (3-seed mean).

## Results (prior weighted F1, t>=1, no x_t)

| Dataset | Weighted F1 | Macro F1 | Accuracy |
| --- | --- | --- | --- |
| IEMOCAP | 0.4714 | 0.4721 | 0.4789 |

(3-seed mean; per-seed and code to be added.)
