# Model Card — Mood Machine

## System Overview

**Mood Machine** is a dual-model text sentiment classifier that predicts whether a short piece of text is *positive*, *negative*, *neutral*, or *mixed*. It combines a rule-based keyword scorer (`mood_analyzer.py`) with a bag-of-words logistic regression model (`ml_experiments.py`) and wraps every prediction with a reliability layer that reports confidence level, edge-case flags, and model agreement.

---

## 1. Model Overview

**Model type:** Both the rule-based model and the ML model were implemented and run side by side on every input.

**Intended purpose:** Classify short, informal text — such as social media posts — into one of four sentiment categories: positive, negative, neutral, or mixed.

**How it works:**
- *Rule-based model:* Text is lowercased, punctuation is removed, and emojis are mapped to text equivalents. Each word is matched against a fixed list of 10 positive and 10 negative keywords. Negation is handled by flipping the score of a keyword if the word immediately before it is a negation word (e.g., "not", "never"). The final numeric score determines the label.
- *ML model:* The same labeled examples from `dataset.py` are converted into bag-of-words vectors using `CountVectorizer`. A logistic regression classifier is trained on those vectors and produces a probability distribution over the four labels, from which the highest-probability label is selected.

---

## 2. Data

**Dataset description:** The dataset consists of 14 short, social-media-style posts stored in `SAMPLE_POSTS`, each with a human-assigned ground-truth label in `TRUE_LABELS`.

**Labeling process:** Labels were assigned by hand based on the overall tone of each post. Posts expressing clear sentiment were straightforward to label. Posts with mixed or ambiguous feelings — such as "Lowkey stressed but kind of proud of myself" — required judgment calls and could reasonably carry more than one valid label.

**Important characteristics:**
- Contains emojis and informal punctuation
- Includes at least one sarcastic example
- Includes posts expressing genuinely mixed emotions
- Short length (typically one sentence)

**Possible issues:** The dataset is small (14 examples), not balanced across all four classes, and drawn from a single informal writing style. It does not represent regional dialects, formal writing, or non-English text.

---

## 3. How the Rule-Based Model Works

**Scoring rules:**
- Each positive keyword match adds +1 to the score; each negative keyword match subtracts 1.
- If the word immediately before a keyword is a negation word, the keyword's contribution is flipped.
- Emojis are converted to descriptive tokens before scoring.
- Labels are assigned by thresholds: score > 0 → positive, score < 0 → negative, score == 0 → neutral.

**Strengths:** Fully transparent and deterministic. Any prediction can be traced back to a specific word match. Fast and requires no training data.

**Weaknesses:** Cannot handle context, sarcasm, subtle phrasing, or any vocabulary outside the fixed keyword list. Multi-word negations and complex sentence structures are not modeled.

---

## 4. How the ML Model Works

**Features used:** Bag-of-words representation via `CountVectorizer` — each unique word in the training set becomes a feature dimension.

**Training data:** Trained on `SAMPLE_POSTS` with labels from `TRUE_LABELS`.

**Training behavior:** Adding the sarcasm-labeled example with a `negative` label caused the ML model to correctly classify similar sarcastic inputs that the rule-based model mislabeled. This demonstrated how directly data quality shapes model behavior.

**Strengths and weaknesses:** The ML model can learn statistical patterns without hand-crafted rules. However, with only 14 training examples, it is prone to overfitting — it partially memorizes the training set rather than learning generalizable patterns. Its reported accuracy is therefore optimistic.

---

## 5. Evaluation

**How the model was evaluated:** Both models were evaluated on the same 14 labeled examples in `dataset.py`. The rule-based model's accuracy was computed directly; the ML model's accuracy was measured using the same set it was trained on (no held-out test set).

**Examples of correct predictions:**
- "I love this class so much" → `positive` (rule-based matches `love`; ML agrees)
- "This is the worst day ever" → `negative` (rule-based matches `worst`; ML agrees)
- "I am not happy about this at all" → `negative` (negation flips `happy`; ML agrees)

**Examples of incorrect predictions:**
- "I absolutely love getting stuck in traffic :)" → predicted `positive` by the rule-based model; true sentiment is sarcastic/negative. The keyword `love` dominates the score and the sarcasm is missed.
- "Lowkey stressed but kind of proud of myself" → predicted `negative` by the rule-based model, `mixed` by the ML model; neither perfectly matches the nuanced true label of `mixed`.

---

## 6. Limitations

- **Vocabulary coverage:** The rule-based model can only detect sentiment expressed through its 20 fixed keywords. Slang, domain-specific language, and informal expressions outside the list are ignored.
- **Small and homogeneous training data:** The ML model was trained on 14 examples from a single style of writing, making it likely to fail on text that differs from those examples.
- **Shallow negation handling:** Only the single word immediately before a keyword is checked for negation. Longer negation spans are not handled.
- **No held-out test set:** Accuracy figures reflect performance on the training data and are therefore inflated relative to real-world performance.
- **Sarcasm detection is heuristic:** The sarcasm flag is triggered by simple surface patterns and will miss most real-world sarcasm.

---

## 7. Ethical Considerations

Mood classification makes automated judgments about human emotional expression, which carries several risks:

- **Consequential misclassification:** A message expressing distress could be labeled `neutral` if the language does not match the keyword list, potentially causing harm in a system designed to escalate such messages.
- **Language bias:** The system was built for informal English. It will perform poorly — and unpredictably — on messages in other languages, regional dialects, or culturally specific slang.
- **Misuse potential:** A sentiment classifier could be used to make automated decisions about people (e.g., screening job applicants, profiling users) without their knowledge. Given its limited accuracy, such use would be both technically unreliable and ethically problematic.

Mitigations built into this system include explicit confidence levels, edge-case warnings, and model-disagreement flags — all of which communicate uncertainty rather than projecting false confidence.

---

## 8. Ideas for Improvement

- Add more labeled examples, particularly for underrepresented classes (`mixed`) and edge cases (sarcasm, negation)
- Introduce a separate held-out validation set to get an honest accuracy estimate
- Replace `CountVectorizer` with TF-IDF to reduce the influence of common words
- Extend the negation window beyond one word to handle phrases like "I don't think this is bad"
- Expand the keyword list with common slang and domain-specific terms
- Integrate a small pre-trained model (e.g., a fine-tuned DistilBERT) as a third comparison signal for high-stakes inputs

---

## 9. Reflection

### Limitations and Biases

The most significant limitation is the small, homogeneous dataset. Fourteen examples drawn from one writing style cannot represent the full diversity of human expression. The fixed keyword vocabulary introduces lexical bias — any sentiment not expressed through those 20 words is invisible to the rule-based model. The shallow negation rule and heuristic sarcasm detection further narrow the range of text the system handles correctly.

### Could the AI Be Misused, and How Would You Prevent It?

Yes. A mood classifier could be misused to make automated judgments about individuals — screening applicants, profiling users, or flagging messages — without accounting for the system's known inaccuracies and biases. Prevention requires: (1) surfacing uncertainty explicitly on every prediction, as this system does through confidence levels and disagreement flags; (2) requiring human review before any prediction informs a real decision; (3) clearly disclosing the system's scope and limitations to anyone who interacts with its output; and (4) validating the system on a representative held-out dataset before any deployment beyond a learning environment.

### What Surprised Me During Reliability Testing

The most striking observation was how quickly the rule-based model fails on sarcasm. "I absolutely love getting stuck in traffic :)" receives a confident `positive` label because `love` is in the positive keyword list — the surrounding context is entirely ignored. What made this more interesting is that the ML model, trained on only 14 examples, correctly labeled a similar sarcastic input as `negative` because it had seen a labeled sarcasm example during training. A second surprise was the value of writing tests that document *known failures* rather than only correct behavior. Formalizing what the system cannot do turned out to be as informative as measuring what it gets right.

### Collaboration with AI During This Project

AI assistance was used throughout the project for code review, debugging, test design, and documentation.

**A helpful contribution.** The most valuable AI suggestion came during the ML model evaluation phase. The AI identified that the model was being evaluated on the same data it had been trained on — a classic overfitting scenario — and explained clearly why the resulting accuracy figure was misleading. With only 14 examples and no held-out validation set, the ML model was partially memorizing labels rather than learning generalizable patterns. This insight directly shaped the decision to document the limitation explicitly in the README, the test suite, and in this model card, and it influenced how the system communicates accuracy to users.

**A flawed suggestion.** During test suite implementation, the AI entered a repetitive loop while generating a particular test case, producing near-identical code blocks multiple times without making forward progress. The session had to be manually interrupted and the generated output discarded. The test case was then written by hand with only targeted AI assistance for specific method signatures. This experience reinforced that AI-generated code requires active supervision — particularly for iterative or sequential tasks where the model can lose track of its own state and regress rather than advance.
