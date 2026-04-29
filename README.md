# Mood Machine

A dual-model sentiment classifier that predicts whether a short piece of text is **positive**, **negative**, **neutral**, or **mixed** — and wraps every prediction with live reliability signals so you know how much to trust the result.

---

## Original Project (Modules 1–3)

This project began as a rule-based mood analyzer built in Modules 1–3 of AI 110. The original goal was to classify short social-media-style posts using a hand-crafted word list and a simple numeric scoring system. The starter system could assign a `positive`, `negative`, or `neutral` label to any text by tallying positive and negative keyword matches. It was deliberately minimal — the interesting work was understanding where the rules break down and why.

---

## Title and Summary

**Mood Machine** is a text sentiment classifier that uses two independent models side by side:

1. **Rule-based model** — keyword scoring with negation handling and emoji preprocessing
2. **ML model** — bag-of-words + logistic regression trained on the same labeled examples

Every prediction comes with a confidence level (`NONE` / `LOW` / `MEDIUM` / `HIGH`), edge-case flags (negation, emoji, possible sarcasm), a consistency guarantee, and a live comparison between the two models. The goal is not just to classify text, but to be transparent about uncertainty — which matters any time an automated system makes a judgment about human emotion.

---

## Architecture Overview

```
╔══════════════════════════════════════════════════════════════════════╗
║                         MOOD MACHINE SYSTEM                         ║
╚══════════════════════════════════════════════════════════════════════╝

  ┌─────────────────────────────────────────────────────────────────┐
  │                        DATA LAYER                               │
  │  dataset.py                                                     │
  │  ┌──────────────────┐  ┌───────────────────┐  ┌─────────────┐  │
  │  │  POSITIVE_WORDS  │  │   NEGATIVE_WORDS  │  │ SAMPLE_POSTS│  │
  │  │  (10 words)      │  │   (10 words)      │  │ TRUE_LABELS │◄─┼── 👤 Human labels
  │  └────────┬─────────┘  └────────┬──────────┘  └──────┬──────┘  │
  └───────────┼────────────────────┼─────────────────────┼─────────┘
              │                    │                      │
              ▼                    ▼                      ▼
  ┌───────────────────────────┐   ┌────────────────────────────────┐
  │    RULE-BASED CLASSIFIER  │   │       ML CLASSIFIER            │
  │    mood_analyzer.py       │   │       ml_experiments.py        │
  │                           │   │                                │
  │  preprocess()             │   │  CountVectorizer               │
  │  (lowercase, emoji map,   │   │  (bag-of-words vectors)        │
  │   remove punct, negation) │   │      +                         │
  │      +                    │   │  LogisticRegression            │
  │  score_text()             │   │  (trained on SAMPLE_POSTS)     │
  │  (word match + negation)  │   │                                │
  │      +                    │   └───────────────┬────────────────┘
  │  predict_label()          │                   │
  └─────────────┬─────────────┘                   │
                └──────────────┬──────────────────┘
                               ▼
  ┌────────────────────────────────────────────────────────────────┐
  │                    RELIABILITY LAYER  (reliability.py)         │
  │                                                                │
  │  EdgeCaseDetector  ·  ConsistencyChecker  ·  ModelComparator   │
  │  confidence_label()  ·  ReliabilityReport                      │
  └──────────────────────┬─────────────────────────────────────────┘
                         ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                     ORCHESTRATOR  (main.py)                     │
  │  startup self-test → dataset eval → batch demo → interactive    │
  │                                                                 │
  │  Automated testing                Human-in-the-loop             │
  │  test_mood_classifier.py          Interactive terminal loop     │
  │  (9 test classes, 60+ tests)      👤 user types → live output   │
  └─────────────────────────────────────────────────────────────────┘
```

**Data flow in plain English:**

1. Text enters — either from `SAMPLE_POSTS` (batch mode) or typed by a user (interactive mode).
2. Both the rule-based model and the ML model produce an independent label.
3. The reliability layer wraps those labels with confidence, edge-case flags, and a model-agreement check.
4. The orchestrator (`main.py`) ties everything together: it runs a startup self-test on launch, evaluates the full dataset, then starts the interactive loop.

**Where humans are involved:**

| Touchpoint | Role |
|---|---|
| `dataset.py` `TRUE_LABELS` | A human labeled every example; these ground-truth labels drive accuracy measurement |
| `test_mood_classifier.py` | A developer wrote 60+ assertions defining what correct behavior looks like |
| Interactive loop | An end user types free text and reads the reliability output in real time |

---

## Setup Instructions

**Prerequisites:** Python 3.10 or later.

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd applied-ai-system-project

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full application
#    (self-test → dataset evaluation → batch demo → interactive loop)
python main.py

# 5. Run only the test suite
python main.py --test
# or with pytest for a richer output
python -m pytest test_mood_classifier.py -v

# 6. Run ML experiments standalone
python ml_experiments.py
```

**Expected first-run output:**

```
[self-test] OK

=== Rule-Based Evaluation on SAMPLE_POSTS ===
  ✓ "I love this class so much"
      predicted=positive (LOW)  true=positive
  ...

=== Interactive Mood Machine ===
Type a sentence to see its predicted mood + reliability info.
Type 'quit' or press Enter to exit.

You:
```

---

## Demo Walkthrough

> **Video walkthrough:** *(Loom link coming soon)*

The text walkthroughs below show the system running end-to-end on four inputs that cover the main cases: clear positive sentiment, negation flipping, sarcasm causing model disagreement, and genuinely mixed feelings.

---

## Sample Interactions

### Example 1 — Clear positive sentiment

```
You: I had such an amazing day today

  Mood       : positive  [confidence: LOW | score: +1]
  Models     : rule=positive  ML=positive  [AGREE]
```

Both models agree. Confidence is `LOW` because only one keyword (`amazing`) matched — a single word is not very strong evidence.

---

### Example 2 — Negation flipping sentiment

```
You: I am not happy about this at all

  Mood       : negative  [confidence: LOW | score: -1]
  Edge cases : negation
  Models     : rule=negative  ML=negative  [AGREE]
```

The word `happy` would normally score +1, but the preceding `not` flips it to −1. The `EdgeCaseDetector` flags the negation so the user knows the result required that extra reasoning step.

---

### Example 3 — Sarcasm, where the rule-based model is unreliable

```
You: I absolutely love getting stuck in traffic :)

  Mood       : positive  [confidence: LOW | score: +1]
  Edge cases : emoji, possible sarcasm
  [!] Sarcasm hint: rule-based score may be unreliable here
  Models     : rule=positive  ML=negative  [DISAGREE]
  [!] Models disagree — consider both predictions
```

The rule-based model scores `love` as +1 and returns `positive`. The ML model, trained on a sarcasm-labeled example in the dataset, correctly returns `negative`. The system surfaces the disagreement explicitly rather than silently picking one answer — because in a real application you would want a human to check this case.

---

### Example 4 — Ambiguous / mixed feelings

```
You: Lowkey stressed but kind of proud of myself

  Mood       : negative  [confidence: LOW | score: -1]
  Edge cases : negation
  Models     : rule=negative  ML=mixed  [DISAGREE]
  [!] Models disagree — consider both predictions
```

The true label for this post is `mixed`. Neither model nails it cleanly, and the disagreement flag correctly signals that this is a hard case. The `stressed` keyword pulls the rule-based score negative; the ML model, having seen a `mixed` example during training, hedges toward that label.

---

## Design Decisions

### Why two models instead of one?

A single model can be confidently wrong. Running a rule-based model and an ML model in parallel means disagreements become a useful signal: when they agree you can be more confident; when they disagree you know to be skeptical. This pattern — using model disagreement as an uncertainty flag — is common in production ML systems.

### Why keyword scoring instead of a pre-trained model like BERT?

The rule-based model is fully transparent: you can read the word list and trace exactly why any prediction was made. That interpretability is more valuable for a learning environment (and for debugging) than raw accuracy. The trade-off is that it misses context, sarcasm, and anything not in the word list.

### Why a reliability layer instead of just returning a label?

Labels without context are misleading. A model that says "positive" about a sarcastic statement is actively harmful if the user trusts it blindly. Surfacing confidence, edge-case flags, and model disagreement forces the system to be honest about what it does not know.

### Trade-offs made

| Decision | Benefit | Cost |
|---|---|---|
| Rule-based + ML side by side | Disagreement = useful uncertainty signal | Two models to maintain |
| Keyword list in `dataset.py` | Easy to inspect and extend | Limited vocabulary, misses slang |
| Negation by single preceding word | Handles "not happy", "never good" | Misses "I don't think this is bad" |
| Training and test set are the same | Keeps the lab self-contained | Accuracy numbers are optimistic (no held-out test set) |
| Heuristic sarcasm detection | Catches obvious cases like `love :)` | False positives; misses dry sarcasm |

---

## Testing Summary

The test suite in `test_mood_classifier.py` covers nine areas with over 60 individual assertions:

| Test Class | What it checks | Result |
|---|---|---|
| `TestPreprocess` | Lowercase, emoji preservation, punctuation removal, repeated-char normalization | Passes |
| `TestScoreText` | Positive/negative scoring, negation flipping, accumulation, cancellation | Passes |
| `TestPredictLabel` | Correct labels for clear cases, negation shifts, custom word lists | Passes |
| `TestEdgeCases` | Emoji detection, negation detection, sarcasm heuristic, crash safety | Passes |
| `TestConsistency` | Same input always returns same output across 5–10 repeated calls | Passes |
| `TestDatasetAccuracy` | Accuracy > 0, confusion matrix shape, correct predictions on obvious examples | Passes |
| `TestMLModelComparison` | ML predictions are valid labels, agreement rate in [0,1], dataset-level metrics | Passes |
| `TestConfidenceLabel` | Correct tier for scores 0 through 10 | Passes |
| `TestReliabilityReport` | Returns expected dict shape, edge-cases list, perfect accuracy on custom data | Passes |

**What worked well:** Testing edge cases in isolation (negation, emoji, sarcasm) made it easy to locate exactly where the rule-based model fails. The `ConsistencyChecker` confirmed that the rule-based model is deterministic, which is not guaranteed for ML models.

**What did not work well:** The ML model is trained and evaluated on the same small dataset (14 examples), so its accuracy number is inflated — it is partially memorizing the training set rather than generalizing. On genuinely new text, it often disagrees with the rule-based model on ambiguous cases.

**Key learning:** Testing an AI system is different from testing regular software. A function either returns the right integer or it does not; a classifier can be "mostly right" in ways that obscure systematic failures. Writing tests that document *known limitations* (like `test_sarcasm_rule_based_known_limitation`) turned out to be just as valuable as tests for correct behavior.

---

## Reflection

Building this project made three things concrete that had previously felt abstract:

**1. Data shapes the model more than the algorithm does.** When I added sarcastic examples to the dataset with correct `negative` labels, the ML model started getting those right. The rule-based model did not improve at all — because rules are static. The lesson: in real ML, the fastest path to better performance is usually better data, not a fancier model.

**2. Transparency is a design choice, not a bonus feature.** It would have been simpler to return just a label. Adding confidence levels, edge-case flags, and model disagreement required extra code, extra tests, and extra thought. But without that layer, the system would present uncertain guesses with the same confidence as clear-cut ones. Any AI system making judgments about people — mood, intent, risk — needs to communicate its uncertainty explicitly.

**3. Edge cases reveal what the model actually learned.** The sarcasm examples ("I love getting stuck in traffic :)") expose that the rule-based model learned "love → positive" rather than learning anything about sentiment. The ML model learned the same shortcut from the training data, just statistically. Neither model understands language; they both pattern-match. Knowing that is important before deciding how much to trust them.
