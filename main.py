"""
Entry point for the Mood Machine rule-based mood analyzer.

Usage:
  python main.py           — evaluate on dataset, then run interactive mode
  python main.py --test    — run the full test suite and exit
"""

import sys
import unittest
from typing import List

from mood_analyzer import MoodAnalyzer
from dataset import SAMPLE_POSTS, TRUE_LABELS
from reliability import (
    EdgeCaseDetector,
    ConsistencyChecker,
    ModelComparator,
    ReliabilityReport,
    confidence_label,
)


# ---------------------------------------------------------------------------
# Startup self-test — runs before anything else to verify the system works
# ---------------------------------------------------------------------------

def run_startup_selftest() -> bool:
    """Quick sanity check: verify predict_label returns the expected labels."""
    a = MoodAnalyzer()
    checks = [
        ("I love this", "positive"),
        ("This is terrible", "negative"),
        ("This is a sentence", "neutral"),
    ]
    passed = sum(1 for text, expected in checks if a.predict_label(text) == expected)
    ok = passed == len(checks)
    status = "OK" if ok else f"WARN — only {passed}/{len(checks)} sanity checks passed"
    print(f"[self-test] {status}")
    return ok


# ---------------------------------------------------------------------------
# Dataset evaluation with full reliability report
# ---------------------------------------------------------------------------

def evaluate_rule_based(posts: List[str], labels: List[str]) -> float:
    """
    Evaluate the rule-based MoodAnalyzer on a labeled dataset.

    Prints each prediction vs. ground truth, then shows the full reliability
    report (accuracy, confusion matrix, per-class metrics, edge cases).
    """
    analyzer = MoodAnalyzer()
    correct = 0

    print("\n=== Rule-Based Evaluation on SAMPLE_POSTS ===")
    for text, true_label in zip(posts, labels):
        predicted = analyzer.predict_label(text)
        score = analyzer.score_text(text)
        conf = confidence_label(score)
        is_correct = predicted == true_label
        if is_correct:
            correct += 1
        mark = "✓" if is_correct else "✗"
        print(f'  {mark} "{text}"')
        print(f'      predicted={predicted} ({conf})  true={true_label}')

    if not posts:
        print("  No labeled examples to evaluate.")
        return 0.0

    accuracy = correct / len(posts)
    print(f"\nRule-based accuracy: {accuracy:.1%}  ({correct}/{len(posts)} correct)")

    # Full reliability report: confusion matrix, per-class F1, edge cases
    report = ReliabilityReport(analyzer)
    report.print_report(posts, labels)

    return accuracy


# ---------------------------------------------------------------------------
# Batch demo
# ---------------------------------------------------------------------------

def run_batch_demo() -> None:
    """Run predictions on SAMPLE_POSTS and display confidence + edge case flags."""
    analyzer = MoodAnalyzer()
    detector = EdgeCaseDetector()

    print("\n=== Batch Demo on SAMPLE_POSTS (rule-based) ===")
    for text in SAMPLE_POSTS:
        label = analyzer.predict_label(text)
        score = analyzer.score_text(text)
        conf = confidence_label(score)
        flags = detector.detect(text)
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        print(f'  "{text}" → {label} [{conf}]{flag_str}')


# ---------------------------------------------------------------------------
# Interactive loop — reliability info on every prediction
# ---------------------------------------------------------------------------

def run_interactive_loop() -> None:
    """
    Let the user type sentences and see predictions with live reliability info:
      - Confidence level (derived from score magnitude)
      - Edge cases detected (negation, emoji, possible sarcasm)
      - Rule-based vs ML comparison (AGREE / DISAGREE)

    Session statistics are shown on exit.
    """
    analyzer = MoodAnalyzer()
    detector = EdgeCaseDetector()
    checker = ConsistencyChecker(analyzer.predict_label)

    # Try to wire up the ML comparator; gracefully skip if unavailable.
    comparator = None
    try:
        from ml_experiments import train_ml_model
        vectorizer, ml_model = train_ml_model(SAMPLE_POSTS, TRUE_LABELS)
        comparator = ModelComparator(analyzer, vectorizer, ml_model)
    except Exception:
        pass

    total = 0
    edge_count = 0
    disagree_count = 0

    print("\n=== Interactive Mood Machine ===")
    print("Type a sentence to see its predicted mood + reliability info.")
    print("Type 'quit' or press Enter to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input == "" or user_input.lower() == "quit":
            break

        total += 1
        label = analyzer.predict_label(user_input)
        score = analyzer.score_text(user_input)
        conf = confidence_label(score)

        print(f"  Mood       : {label}  [confidence: {conf} | score: {score:+d}]")

        # Consistency check (deterministic — always passes for a pure function,
        # but documents the guarantee explicitly to the user)
        consistent, _ = checker.check(user_input)
        if not consistent:               # should never happen
            print("  [!!] WARNING: inconsistent predictions detected")

        # Edge case detection
        flags = detector.detect(user_input)
        if flags:
            edge_count += 1
            print(f"  Edge cases : {', '.join(flags)}")
            if "possible sarcasm" in flags:
                print("  [!] Sarcasm hint: rule-based score may be unreliable here")

        # Model comparison
        if comparator:
            cmp = comparator.compare(user_input)
            agreement = "AGREE" if cmp["agree"] else "DISAGREE"
            print(f"  Models     : rule={cmp['rule_based']}  ML={cmp['ml_model']}  [{agreement}]")
            if not cmp["agree"]:
                disagree_count += 1
                print("  [!] Models disagree — consider both predictions")

        print()

    # Session summary
    if total > 0:
        print("--- Session Summary ---")
        print(f"  Predictions made : {total}")
        print(f"  Edge cases found : {edge_count}")
        if comparator and total > 0:
            agree_rate = (total - disagree_count) / total
            print(f"  Model agreement  : {agree_rate:.1%}")
    print("Goodbye from the Mood Machine.")


# ---------------------------------------------------------------------------
# Test-suite runner (invoked by --test flag)
# ---------------------------------------------------------------------------

def _run_tests() -> None:
    import test_mood_classifier
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_mood_classifier)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--test" in sys.argv:
        _run_tests()

    run_startup_selftest()
    evaluate_rule_based(SAMPLE_POSTS, TRUE_LABELS)
    run_batch_demo()
    run_interactive_loop()
