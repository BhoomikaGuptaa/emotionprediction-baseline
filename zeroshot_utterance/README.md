# Zero-shot Utterance Prediction (IEMOCAP)

**Task:** given the dialogue history (turns 0..t-1, each with speaker and gold emotion tag) and the name of the next speaker, generate the next utterance. The model never sees turn t. Generations are scored against the real utterance with similarity metrics. Same 1592 causal test points as the emotion-prediction task (the script asserts the point set matches the emotion loader exactly).

This is the zero-shot floor for the utterance half of the pipeline: a trained generate-then-predict model should beat everything in this folder.

## Results (full IEMOCAP test, 1592 points)

| Metric | Parrot floor | Zero-shot Qwen2.5-7B | OpenAI (gpt-4o-mini) |
| --- | --- | --- | --- |
| BLEU | 1.56 | 0.70 | |
| chrF | 12.91 | 10.84 | |
| ROUGE-1 | 0.120 | 0.111 | |
| ROUGE-2 | 0.027 | 0.018 | |
| ROUGE-L | 0.105 | 0.100 | |
| METEOR | 0.061 | 0.046 | |
| BERTScore-F1 | 0.103 | 0.117 | |
| SBERT cosine | 0.236 | 0.210 | |

Headline: the parrot floor beats the zero-shot 7B on 7 of 8 similarity metrics.

## What is the parrot, and why is it here?

The parrot is a no-model baseline. Its "prediction" for the next utterance is a word-for-word copy of the PREVIOUS utterance. One line of logic, no GPU:

    prediction = history[-1]

That sounds dumb on purpose. It is here for the same reason the persistence floor is in the emotion task ("predict the last emotion again", which scored 0.7323 weighted F1 and beat every trained model there). Both floors measure the same phenomenon: conversational inertia. Consecutive turns in real dialogue share a lot. People echo words back, confirm, repeat names, stay on topic with the same vocabulary. So the previous utterance already overlaps heavily with the next one, in both words and meaning.

Similarity metrics measure exactly that overlap. BLEU, ROUGE and chrF count shared words and characters; BERTScore and SBERT measure shared meaning. The parrot collects all of that shared material for free, without generating anything.

Isn't a same-word match meaningless? No, and that is the point. The metrics do not know or care where the words came from. If a copy of the last line scores HIGHER than a 7B model's genuinely sensible new utterance, that tells you two real things:

1. Zero-shot generation lands far from the ground truth. The 7B's outputs are plausible ("I understand your frustration, sir. Let's get this sorted out...") but plausible-and-new drifts further from what the person actually said than a verbatim echo does. Many valid continuations exist; the model rarely produces the one that happened.
2. Any trained model must clear this bar. If a model trained on similarity rewards cannot out-similarity a copy-paste of the last line, the training added nothing measurable. The parrot makes that test explicit and free.

So the parrot is not a serious method. It is a control, the generation-task twin of the persistence floor. Reporting it keeps every other number in this table honest.

## Folders

- `qwen_7b/` - zero-shot local model (Qwen2.5-7B-Instruct)
- `parrot_floor/` - repeat-last-utterance control
- `openai_api/` - same experiment through the OpenAI API (gpt-4o-mini)
- `shared/` - the IEMOCAP loader shared with the emotion baselines (guarantees the identical point set)

## Settings (all methods)

History window 12 turns, past gold emotion tags shown in the prompt (labels_in_history=1, matching the project H_t protocol), greedy / temperature-0 decoding, max 64 new tokens. Scored with sacrebleu (BLEU, chrF), rouge_score, NLTK METEOR, BERTScore (roberta-large), SBERT (all-mpnet-base-v2). transformers pinned <4.46: newer versions changed BERTScore values, so all comparisons must be scored under the same pinned environment. Any trained model evaluated against these floors must use the same labels_in_history setting.
