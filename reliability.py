"""
Reliability system for the Mood Machine classifier.

Provides edge-case detection, consistency checking, model comparison,
and accuracy metrics that are wired into the main application so that
every prediction includes reliability context.
"""

import re
from typing import Callable, Dict, List, Tuple

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Words that negate the sentiment of the following token.
_NEGATION_WORDS = {"not", "never", "no", "dont", "doesnt", "didnt", "wont", "cant", "couldnt"}

# Matches text-based emoticons like :) :( ;) :-( :D
_TEXT_EMOJI_RE = re.compile(r'[:;]-?[)(DdPp]')

# Matches common Unicode emoji ranges
_UNICODE_EMOJI_RE = re.compile(
    r'[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF]',
    re.UNICODE,
)

# Positive words that appear in sarcastic phrases paired with a smiley
_SARCASM_POSITIVES = {"love", "amazing", "great", "awesome", "wonderful", "best", "fantastic"}


class EdgeCaseDetector:
    """Detects text patterns that challenge sentiment classifiers."""

    def has_negation(self, text: str) -> bool:
        tokens = re.sub(r"[^\w\s]", "", text.lower()).split()
        return any(t in _NEGATION_WORDS for t in tokens)

    def has_emoji(self, text: str) -> bool:
        return bool(_TEXT_EMOJI_RE.search(text) or _UNICODE_EMOJI_RE.search(text))

    def has_sarcasm_signal(self, text: str) -> bool:
        """Heuristic: positive word + smiley face often signals sarcasm."""
        lower = text.lower()
        has_positive = any(w in lower.split() for w in _SARCASM_POSITIVES)
        has_smiley = bool(re.search(r'[:;]-?\)', text))
        return has_positive and has_smiley

    def detect(self, text: str) -> List[str]:
        """Return a list of edge-case type strings detected in *text*."""
        flags: List[str] = []
        if self.has_negation(text):
            flags.append("negation")
        if self.has_emoji(text):
            flags.append("emoji")
        if self.has_sarcasm_signal(text):
            flags.append("possible sarcasm")
        return flags


class ConsistencyChecker:
    """Verifies that a predictor returns the same label for the same input every time."""

    def __init__(self, predict_fn: Callable[[str], str]) -> None:
        self._predict = predict_fn

    def check(self, text: str, runs: int = 5) -> Tuple[bool, List[str]]:
        """Run *predict_fn* `runs` times; return (is_consistent, all_predictions)."""
        predictions = [self._predict(text) for _ in range(runs)]
        return len(set(predictions)) == 1, predictions


class ModelComparator:
    """Compares rule-based and ML model predictions side by side."""

    def __init__(self, analyzer, vectorizer, ml_model) -> None:
        self.analyzer = analyzer
        self.vectorizer = vectorizer
        self.ml_model = ml_model

    def compare(self, text: str) -> Dict:
        """Return a dict with both predictions and whether they agree."""
        rule_pred = self.analyzer.predict_label(text)
        ml_pred = self.ml_model.predict(self.vectorizer.transform([text]))[0]
        return {
            "rule_based": rule_pred,
            "ml_model": ml_pred,
            "agree": rule_pred == ml_pred,
        }

    def compare_on_dataset(self, texts: List[str], labels: List[str]) -> Dict:
        """Compare both models against ground-truth labels; return summary metrics."""
        rule_preds = [self.analyzer.predict_label(t) for t in texts]
        ml_preds = list(self.ml_model.predict(self.vectorizer.transform(texts)))
        agreement_rate = sum(r == m for r, m in zip(rule_preds, ml_preds)) / len(texts)
        return {
            "rule_accuracy": accuracy_score(labels, rule_preds),
            "ml_accuracy": accuracy_score(labels, ml_preds),
            "agreement_rate": agreement_rate,
            "rule_predictions": rule_preds,
            "ml_predictions": ml_preds,
        }


def confidence_label(score: int) -> str:
    """Map a numeric mood score to a human-readable confidence level."""
    magnitude = abs(score)
    if magnitude == 0:
        return "NONE"
    if magnitude == 1:
        return "LOW"
    if magnitude == 2:
        return "MEDIUM"
    return "HIGH"


class ReliabilityReport:
    """Generates full reliability metrics for a MoodAnalyzer instance."""

    def __init__(self, analyzer) -> None:
        self.analyzer = analyzer
        self.edge_detector = EdgeCaseDetector()

    def evaluate(self, texts: List[str], labels: List[str]) -> Dict:
        """Compute accuracy, confusion matrix, per-class metrics, and edge cases."""
        preds = [self.analyzer.predict_label(t) for t in texts]
        all_labels = sorted(set(labels) | set(preds))
        return {
            "accuracy": accuracy_score(labels, preds),
            "confusion_matrix": confusion_matrix(labels, preds, labels=all_labels),
            "label_order": all_labels,
            "classification_report": classification_report(
                labels, preds, labels=all_labels, zero_division=0
            ),
            "edge_cases": [
                (t, self.edge_detector.detect(t))
                for t in texts
                if self.edge_detector.detect(t)
            ],
            "predictions": preds,
        }

    def print_report(self, texts: List[str], labels: List[str]) -> None:
        result = self.evaluate(texts, labels)
        divider = "=" * 52
        print(f"\n{divider}")
        print("  RELIABILITY REPORT — Rule-Based MoodAnalyzer")
        print(divider)
        print(f"  Dataset size : {len(texts)} examples")
        print(f"  Accuracy     : {result['accuracy']:.1%}")
        print(f"\n  Label order  : {result['label_order']}")
        print("\n  Confusion Matrix:")
        for line in _format_confusion_matrix(
            result["confusion_matrix"], result["label_order"]
        ).splitlines():
            print(f"    {line}")
        print("\n  Per-Class Metrics:")
        for line in result["classification_report"].splitlines():
            print(f"    {line}")
        edge = result["edge_cases"]
        print(f"\n  Edge Cases in Dataset ({len(edge)} found):")
        for text, flags in edge:
            print(f"    [{', '.join(flags)}]  {text!r}")
        print(divider)


def _format_confusion_matrix(cm, labels: List[str]) -> str:
    col_w = max(len(l) for l in labels) + 2
    header = " " * col_w + "".join(l.center(col_w) for l in labels)
    rows = [header, "-" * len(header)]
    for i, label in enumerate(labels):
        row = label.ljust(col_w) + "".join(str(v).center(col_w) for v in cm[i])
        rows.append(row)
    return "\n".join(rows)
