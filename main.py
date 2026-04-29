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
from logger import setup_logger, log_prediction, log_edge_case_failure, log_error


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
    logger = setup_logger()

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
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() == "quit":
            break

        # --- Input validation ---
        if not user_input:
            print("  [!] Please type something. (Type 'quit' to exit.)\n")
            log_error(logger, error_type="empty_input")
            continue

        if len(user_input) > 500:
            print(f"  [!] Input too long ({len(user_input)} chars). Please keep it under 500 characters.\n")
            log_error(logger, error_type="input_too_long", detail=f"{len(user_input)} chars")
            continue

        if not any(c.isalpha() for c in user_input):
            print(f"  [!] Input must contain at least one word. Got: {user_input!r}\n")
            log_error(logger, error_type="no_alpha_chars", detail=user_input)
            continue

        # --- Prediction ---
        ml_pred = None
        models_agree = None
        flags: List[str] = []

        try:
            label = analyzer.predict_label(user_input)
            score = analyzer.score_text(user_input)
            conf = confidence_label(score)
        except Exception as exc:
            print(f"  [!] Prediction failed: {exc}\n")
            log_error(logger, error_type="prediction_error", detail=str(exc))
            continue

        total += 1
        print(f"  Mood       : {label}  [confidence: {conf} | score: {score:+d}]")

        # Consistency check (deterministic — always passes for a pure function,
        # but documents the guarantee explicitly to the user)
        consistent, _ = checker.check(user_input)
        if not consistent:
            print("  [!!] WARNING: inconsistent predictions detected")
            log_error(logger, error_type="inconsistent_prediction", detail=user_input)

        # Edge case detection
        flags = detector.detect(user_input)
        if flags:
            edge_count += 1
            print(f"  Edge cases : {', '.join(flags)}")
            log_edge_case_failure(logger, text=user_input, flags=flags)
            if "possible sarcasm" in flags:
                print("  [!] Sarcasm hint: rule-based score may be unreliable here")

        # Model comparison
        if comparator:
            try:
                cmp = comparator.compare(user_input)
                ml_pred = cmp["ml_model"]
                models_agree = cmp["agree"]
                agreement = "AGREE" if models_agree else "DISAGREE"
                print(f"  Models     : rule={cmp['rule_based']}  ML={ml_pred}  [{agreement}]")
                if not models_agree:
                    disagree_count += 1
                    print("  [!] Models disagree — consider both predictions")
            except Exception as exc:
                print(f"  [!] ML comparison failed: {exc}")
                log_error(logger, error_type="ml_comparison_error", detail=str(exc))

        log_prediction(
            logger,
            text=user_input,
            rule_pred=label,
            ml_pred=ml_pred,
            confidence=conf,
            score=score,
            edge_cases=flags,
            models_agree=models_agree,
        )

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
