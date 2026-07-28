# Zero-shot Utterance Prediction (IEMOCAP)

**Task:** given the dialogue history (turns 0..t-1, each with speaker and gold emotion tag) and the name of the next speaker, generate the next utterance. The model never sees turn t. Generations are scored against the real utterance with similarity metrics. Same 1592 causal test points as the emotion-prediction task.

This is the zero-shot floor for the utterance half of the pipeline: a trained generate-then-predict model should beat everything in this folder.

## Results (full IEMOCAP test, 1592 points)

| Metric | Parrot floor | Zero-shot Qwen2.5-7B | Zero-shot GPT-4o-mini | Dialogue-sim Qwen2.5-7B | Dialogue-sim Llama-3.2-3B( original repo used this) |
| --- | --- | --- | --- | --- | --- |
| BLEU | **1.56** | 0.70 | 0.83 | 0.77 | 1.06 |
| chrF | 12.91 | 10.84 | 12.01 | **15.73** | 15.31 |
| ROUGE-1 | 0.120 | 0.111 | 0.125 | 0.123 | **0.128** |
| ROUGE-2 | **0.027** | 0.018 | 0.022 | 0.019 | 0.023 |
| ROUGE-L | 0.105 | 0.100 | **0.113** | 0.103 | 0.109 |
| METEOR | 0.061 | 0.046 | 0.056 | 0.066 | **0.069** |
| BERTScore-F1 | 0.103 | 0.117 | **0.119** | 0.093 | 0.092 |
| SBERT cosine | **0.236** | 0.210 | 0.216 | 0.211 | 0.204 |

Takeaways: the parrot floor still tops SBERT, BLEU and ROUGE-2, so no zero-shot method beats copying the last line on overall semantic similarity. The dialogue-sim think-then-predict recipe wins the lexical metrics (chrF, METEOR, ROUGE-1) but gives up BERTScore versus direct prompting. Under the dialogue-sim recipe the 3B is roughly even with the 7B.

## What is the parrot?

The parrot has no model. Its "prediction" is just a copy of the previous utterance:

    prediction = history[-1]

It works because people in conversation echo each other: they repeat words, confirm, stay on topic. So the last utterance already overlaps a lot with the next one, and similarity metrics reward exactly that overlap.

It is the utterance version of the persistence floor from the emotion task (repeat the last emotion, 0.7323 weighted F1, beat every trained model). It is a control, not a method: if a model cannot beat a copy of the last line, its predictions are not measurably closer to what the person actually said. No zero-shot method here clears it on semantic similarity, which shows how hard this task is and sets the bar for the trained model.

## What is the dialogue-sim inference?

The prompting recipe from kanishkg/dialogue-sim (Learning to Simulate Human Dialogue): the model first reasons step by step inside `<think>` tags, then predicts the next utterance inside `<dialogue>` tags, sampled at temperature 0.3 / top_p 0.95 with a 512 token budget. Their system prompt and sampling settings are used verbatim; only the data (IEMOCAP instead of DailyDialog) and real speaker labels differ. Run with two models: Qwen2.5-7B (matching the zero-shot run, isolates the recipe) and Llama-3.2-3B (their repo's exact default model). No training is involved.

**Note on reproducibility:** BERTScore values change across transformers library versions. All numbers in this table were scored under the same pinned environment (transformers < 4.46). Model outputs in the dialogue-sim runs were post-processed to remove leading speaker-tag artifacts of the prompt format before scoring (uniform rule, disclosed per-folder). Any model evaluated against these floors must use the same labels_in_history setting and the same pin.
