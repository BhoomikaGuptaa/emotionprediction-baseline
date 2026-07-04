# Zero-shot Utterance Prediction (IEMOCAP)

**Task:** given the dialogue history (turns 0..t-1, each with speaker and gold emotion tag) and the name of the next speaker, generate the next utterance. The model never sees turn t. Generations are scored against the real utterance with similarity metrics. Same 1592 causal test points as the emotion-prediction task.

This is the zero-shot floor for the utterance half of the pipeline: a trained generate-then-predict model should beat everything in this folder.

## Results (full IEMOCAP test, 1592 points)

| Metric | Parrot floor | Zero-shot Qwen2.5-7B | OpenAI (gpt-4o-mini) |
| --- | --- | --- | --- |
| BLEU | 1.56 | 0.70 |0.83 |
| chrF | 12.91 | 10.84 |12.01 |
| ROUGE-1 | 0.120 | 0.111 |0.125 |
| ROUGE-2 | 0.027 | 0.018 |0.022 |
| ROUGE-L | 0.105 | 0.100 | 0.113|
| METEOR | 0.061 | 0.046 |0.056 |
| BERTScore-F1 | 0.103 | 0.117 |0.119 |
| SBERT cosine | 0.236 | 0.210 | 0.216|

The parrot floor beats the zero-shot 7B on 7 of 8 similarity metrics.

## What is the parrot?

The parrot has no model. Its "prediction" is just a copy of the previous utterance:

    prediction = history[-1]

It works because people in conversation echo each other: they repeat words, confirm, stay on topic. So the last utterance already overlaps a lot with the next one, and similarity metrics reward exactly that overlap.

It is the utterance version of the persistence floor from the emotion task (repeat the last emotion, 0.7323 weighted F1, beat every trained model). It is a control, not a method: if a model cannot beat a copy of the last line, its predictions are not measurably closer to what the person actually said. The zero-shot 7B loses to it on 7 of 8 metrics, which shows how hard this task is and sets the bar for the trained model.

**Note on reproducibility:** BERTScore values change across transformers library versions. All numbers in this table were scored under the same pinned environment (transformers < 4.46). Any model evaluated against these floors must be scored under the same pin, or the BERTScore column will not be comparable.
