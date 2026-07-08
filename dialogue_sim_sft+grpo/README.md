# Dialogue-Sim: SFT-only Utterance Generation + Downstream Emotion Recognition (IEMOCAP)

Faithful reproduction of kanishkg/dialogue-sim (SFT → GRPO, LLM-as-judge reward)
on IEMOCAP. 

 — GRPO training plateaued (reward
flat across the full run, 0.218→0.216) and showed hallucination on qualitative
review (see GRPO findings below). SFT-only is the stronger generation checkpoint
on every similarity metric.

---

## 1. Utterance generation — similarity metrics (full IEMOCAP test, 1592 points)

| Metric | Parrot floor | Zero-shot Qwen2.5-7B | Zero-shot GPT-4o-mini | Dialogue-sim Qwen2.5-7B | Dialogue-sim Llama-3.2-3B (original repo) | **Dialogue-sim SFT-only (ours, trained)** | Dialogue-sim SFT+GRPO (ours, trained) |
|---|---|---|---|---|---|---|---|
| BLEU | **1.56** | 0.70 | 0.83 | 0.77 | 1.06 | 0.86 | 0.79 |
| chrF | 12.91 | 10.84 | 12.01 | **15.73** | 15.31 | 10.55 | 10.10 |
| ROUGE-1 | 0.120 | 0.111 | 0.125 | 0.123 | **0.128** | 0.121 | 0.115 |
| ROUGE-2 | **0.027** | 0.018 | 0.022 | 0.019 | 0.023 | 0.023 | 0.022 |
| ROUGE-L | 0.105 | 0.100 | **0.113** | 0.103 | 0.109 | 0.110 | 0.105 |
| METEOR | 0.061 | 0.046 | 0.056 | 0.066 | **0.069** | 0.054 | 0.051 |
| BERTScore-F1 | 0.103 | 0.117 | **0.119** | 0.093 | 0.092 | 0.096 | 0.090 |
| SBERT cosine | **0.236** | 0.210 | 0.216 | 0.211 | 0.204 | 0.211 | 0.202 |

---

## 2. Downstream emotion recognition on SFT-predicted utterances

| Model | Weighted F1 | Macro F1 | Accuracy | Parse-fail |
|---|---|---|---|---|
| Llama-3.2-1B-Instruct | 0.3553 | 0.3636 | 0.3926 | 5.4% |
| Llama-3.1-8B-Instruct | 0.3667 | 0.3586 | 0.3637 | 0.1% |
| Qwen2.5-7B-Instruct | 0.4233 | 0.4086 | 0.4259 | 2.7% |
| GPT-4o-mini | 0.4783 | 0.4639 | 0.4793 | 0.0% |

---

## 3. Comparison to blind (history-only) zero-shot — B2

| Model | Blind (B2, history only) | Utterance-conditioned (SFT-predicted) | Δ |
|---|---|---|---|
| Llama-3.2-1B | 0.4150 | 0.3553 | **-0.0597** |
| Llama-3.1-8B | 0.5981 | 0.3667 | **-0.2314** |
| Qwen2.5-3B → 2.5-7B* | 0.5542 | 0.4233 | -0.1309 |
| GPT-4o-mini | — | 0.4783 | no B2 blind row run yet |


---

## GRPO findings 

- Judge reward flat across the full training run: 0.218 (first 10 logged
  steps) → 0.216 (last 10). No meaningful improvement on GRPO's own objective.
- SFT-only beats SFT+GRPO on every similarity metric . 
- Qualitative review of 15 matched points found hallucination in GRPO output
  not present in SFT-only (fabricated character names, invented instruction
  sequences not grounded in context).
- Independently confirmed by the dialogue-sim authors' own paper
  (arXiv:2601.04436): LLM-as-judge GRPO reward-hacks rather than improving
  human-response likelihood; their published repo only implements this judge
  variant, not the log-probability/latent-variable method their paper shows
  actually works (unreleased).
