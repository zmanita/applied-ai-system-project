"""
Comprehensive test suite for the Mood Machine classifier.

Run directly:   python test_mood_classifier.py
With pytest:    python -m pytest test_mood_classifier.py -v

Test classes:
  TestPreprocess          — MoodAnalyzer.preprocess()
  TestScoreText           — MoodAnalyzer.score_text()
  TestPredictLabel        — MoodAnalyzer.predict_label()
  TestEdgeCases           — emojis, negations, sarcasm via EdgeCaseDetector
  TestConsistency         — same input always yields same output
  TestDatasetAccuracy     — rule-based accuracy against TRUE_LABELS
  TestMLModelComparison   — rule-based vs ML model comparison
  TestConfidenceLabel     — confidence_label() utility
  TestReliabilityReport   — ReliabilityReport.evaluate()
"""

import sys
import unittest

from mood_analyzer import MoodAnalyzer
from dataset import SAMPLE_POSTS, TRUE_LABELS
from reliability import (
    EdgeCaseDetector,
    ConsistencyChecker,
    ModelComparator,
    ReliabilityReport,
    confidence_label,
)
from ml_experiments import train_ml_model, predict_single_text


# ---------------------------------------------------------------------------
# MoodAnalyzer.preprocess()
# ---------------------------------------------------------------------------

class TestPreprocess(unittest.TestCase):

    def setUp(self):
        self.a = MoodAnalyzer()

    def test_lowercases_input(self):
        self.assertIn("happy", self.a.preprocess("HAPPY"))

    def test_strips_leading_trailing_whitespace(self):
        self.assertIn("happy", self.a.preprocess("  happy  "))

    def test_removes_ascii_punctuation(self):
        # "happy!" should yield "happy", not "happy!"
        tokens = self.a.preprocess("happy!")
        self.assertIn("happy", tokens)
        self.assertNotIn("happy!", tokens)

    def test_normalizes_repeated_characters(self):
        tokens = self.a.preprocess("soooooo good")
        self.assertIn("soo", tokens)
        self.assertNotIn("soooooo", tokens)

    def test_preserves_smile_text_emoji(self):
        tokens = self.a.preprocess("great day :)")
        self.assertIn(":)", tokens)

    def test_preserves_frown_text_emoji(self):
        tokens = self.a.preprocess("bad day :(")
        self.assertIn(":(", tokens)

    def test_preserves_unicode_emoji(self):
        tokens = self.a.preprocess("so funny 😂")
        self.assertIn("😂", tokens)

    def test_returns_list(self):
        self.assertIsInstance(self.a.preprocess("hello world"), list)

    def test_empty_string_returns_empty_list(self):
        self.assertEqual(self.a.preprocess(""), [])

    def test_whitespace_only_returns_empty_list(self):
        self.assertEqual(self.a.preprocess("   "), [])


# ---------------------------------------------------------------------------
# MoodAnalyzer.score_text()
# ---------------------------------------------------------------------------

class TestScoreText(unittest.TestCase):

    def setUp(self):
        self.a = MoodAnalyzer()

    def test_positive_word_gives_positive_score(self):
        self.assertGreater(self.a.score_text("happy"), 0)

    def test_negative_word_gives_negative_score(self):
        self.assertLess(self.a.score_text("terrible"), 0)

    def test_no_mood_words_scores_zero(self):
        self.assertEqual(self.a.score_text("this is a sentence"), 0)

    def test_not_flips_positive_word(self):
        self.assertLess(self.a.score_text("not happy"), 0)

    def test_not_flips_negative_word(self):
        self.assertGreater(self.a.score_text("not terrible"), 0)

    def test_never_acts_as_negation(self):
        self.assertLess(self.a.score_text("never good"), 0)

    def test_no_acts_as_negation(self):
        self.assertLess(self.a.score_text("no love"), 0)

    def test_multiple_positive_words_accumulate(self):
        self.assertGreater(self.a.score_text("love this amazing day"), 1)

    def test_opposing_words_cancel(self):
        # one positive + one negative → sum to zero
        self.assertEqual(self.a.score_text("happy and sad"), 0)

    def test_returns_int(self):
        self.assertIsInstance(self.a.score_text("happy"), int)

    def test_empty_string_scores_zero(self):
        self.assertEqual(self.a.score_text(""), 0)


# ---------------------------------------------------------------------------
# MoodAnalyzer.predict_label()
# ---------------------------------------------------------------------------

class TestPredictLabel(unittest.TestCase):

    def setUp(self):
        self.a = MoodAnalyzer()
        self.valid = {"positive", "negative", "neutral", "mixed"}

    def test_clearly_positive_text(self):
        self.assertEqual(self.a.predict_label("I love this so much"), "positive")

    def test_clearly_negative_text(self):
        self.assertEqual(self.a.predict_label("Today was terrible and awful"), "negative")

    def test_neutral_text_with_no_mood_words(self):
        self.assertEqual(self.a.predict_label("This is a sentence"), "neutral")

    def test_returns_string(self):
        self.assertIsInstance(self.a.predict_label("anything"), str)

    def test_returns_valid_label(self):
        self.assertIn(self.a.predict_label("I love a terrible day"), self.valid)

    def test_not_none(self):
        self.assertIsNotNone(self.a.predict_label("some text"))

    def test_negation_shifts_label_to_negative(self):
        self.assertEqual(self.a.predict_label("I am not happy about this"), "negative")

    def test_negation_shifts_label_to_positive(self):
        self.assertEqual(self.a.predict_label("not bad at all"), "positive")

    def test_custom_word_lists_respected(self):
        custom = MoodAnalyzer(positive_words=["spectacular"], negative_words=["dreadful"])
        self.assertEqual(custom.predict_label("spectacular view"), "positive")
        self.assertEqual(custom.predict_label("dreadful weather"), "negative")


# ---------------------------------------------------------------------------
# Edge cases: emojis, negations, sarcasm
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):

    def setUp(self):
        self.a = MoodAnalyzer()
        self.d = EdgeCaseDetector()

    # --- Emoji detection ---

    def test_text_smiley_emoji_flagged(self):
        self.assertIn("emoji", self.d.detect("I'm happy :)"))

    def test_text_frown_emoji_flagged(self):
        self.assertIn("emoji", self.d.detect("feeling down :("))

    def test_unicode_emoji_flagged(self):
        self.assertIn("emoji", self.d.detect("This is amazing 😂"))

    def test_no_emoji_not_flagged(self):
        self.assertNotIn("emoji", self.d.detect("I am happy today"))

    def test_sad_emoji_does_not_crash_analyzer(self):
        label = self.a.predict_label("Just vibing :(")
        self.assertIsNotNone(label)

    # --- Negation detection ---

    def test_not_happy_detected_as_negation(self):
        self.assertIn("negation", self.d.detect("I am not happy about this"))

    def test_never_detected_as_negation(self):
        self.assertIn("negation", self.d.detect("I never feel good about this"))

    def test_no_mood_word_detected_as_negation(self):
        self.assertIn("negation", self.d.detect("no good options left"))

    def test_plain_text_not_flagged_for_negation(self):
        self.assertNotIn("negation", self.d.detect("I am happy today"))

    def test_not_bad_scores_positive(self):
        # "not bad" flips the negative word → positive score
        self.assertGreater(self.a.score_text("not bad"), 0)

    def test_not_happy_predicts_negative(self):
        self.assertEqual(self.a.predict_label("I am not happy"), "negative")

    # --- Sarcasm detection ---

    def test_positive_word_plus_smiley_flagged_as_sarcasm(self):
        self.assertIn("possible sarcasm", self.d.detect("I love getting stuck in traffic :)"))

    def test_positive_word_without_smiley_not_sarcasm(self):
        self.assertNotIn("possible sarcasm", self.d.detect("I love getting stuck in traffic"))

    def test_sarcasm_rule_based_known_limitation(self):
        # Rule-based model is expected to score "love" as positive even in sarcastic context.
        # True label is "negative" — this test documents the known limitation, not a bug.
        text = "I love getting stuck in traffic :)"
        label = self.a.predict_label(text)
        self.assertIn(label, {"positive", "negative", "neutral", "mixed"})
        # Check that EdgeCaseDetector correctly flags it
        self.assertIn("possible sarcasm", self.d.detect(text))

    # --- Mixed/complex inputs ---

    def test_mixed_emotions_does_not_crash(self):
        label = self.a.predict_label("Feeling tired but kind of hopeful")
        self.assertIsNotNone(label)

    def test_slang_text_does_not_crash(self):
        label = self.a.predict_label("No cap, this is actually amazing 💀")
        self.assertIsNotNone(label)

    def test_empty_string_does_not_crash(self):
        label = self.a.predict_label("")
        self.assertIsNotNone(label)

    def test_only_emojis_does_not_crash(self):
        label = self.a.predict_label(":) :( 😂")
        self.assertIsNotNone(label)


# ---------------------------------------------------------------------------
# Consistency: same input → same output every time
# ---------------------------------------------------------------------------

class TestConsistency(unittest.TestCase):

    def setUp(self):
        self.a = MoodAnalyzer()
        self.c = ConsistencyChecker(self.a.predict_label)

    def _assert_consistent(self, text: str):
        ok, preds = self.c.check(text)
        self.assertTrue(ok, f"Inconsistent for {text!r}: {preds}")

    def test_positive_text_consistent(self):
        self._assert_consistent("I love this class so much")

    def test_negative_text_consistent(self):
        self._assert_consistent("Today was a terrible day")

    def test_neutral_text_consistent(self):
        self._assert_consistent("This is a sentence")

    def test_negation_consistent(self):
        self._assert_consistent("I am not happy about this")

    def test_emoji_text_consistent(self):
        self._assert_consistent("Great day :)")

    def test_empty_string_consistent(self):
        self._assert_consistent("")

    def test_uppercase_consistent(self):
        self._assert_consistent("HAPPY DAY")

    def test_mixed_case_same_as_lowercase(self):
        # Preprocess lowercases everything, so HAPPY and happy should match
        label_upper = self.a.predict_label("HAPPY")
        label_lower = self.a.predict_label("happy")
        self.assertEqual(label_upper, label_lower)

    def test_repeated_calls_return_same_label(self):
        text = "feeling stressed and upset"
        labels = {self.a.predict_label(text) for _ in range(10)}
        self.assertEqual(len(labels), 1)


# ---------------------------------------------------------------------------
# Dataset accuracy: rule-based model vs TRUE_LABELS
# ---------------------------------------------------------------------------

class TestDatasetAccuracy(unittest.TestCase):

    def setUp(self):
        self.a = MoodAnalyzer()
        self.report = ReliabilityReport(self.a)

    def test_accuracy_above_zero(self):
        result = self.report.evaluate(SAMPLE_POSTS, TRUE_LABELS)
        self.assertGreater(result["accuracy"], 0.0)

    def test_accuracy_is_float_in_range(self):
        result = self.report.evaluate(SAMPLE_POSTS, TRUE_LABELS)
        self.assertIsInstance(result["accuracy"], float)
        self.assertGreaterEqual(result["accuracy"], 0.0)
        self.assertLessEqual(result["accuracy"], 1.0)

    def test_confusion_matrix_is_square(self):
        result = self.report.evaluate(SAMPLE_POSTS, TRUE_LABELS)
        n = len(result["label_order"])
        self.assertEqual(result["confusion_matrix"].shape, (n, n))

    def test_predictions_length_matches_dataset(self):
        result = self.report.evaluate(SAMPLE_POSTS, TRUE_LABELS)
        self.assertEqual(len(result["predictions"]), len(SAMPLE_POSTS))

    def test_clearly_positive_post_classified_correctly(self):
        # "I love this class so much" → love → score +1 → positive
        self.assertEqual(self.a.predict_label("I love this class so much"), "positive")

    def test_clearly_positive_post_2_classified_correctly(self):
        # "So excited for the weekend" → excited → score +1 → positive
        self.assertEqual(self.a.predict_label("So excited for the weekend"), "positive")

    def test_clearly_negative_post_classified_correctly(self):
        # "Today was a terrible day" → terrible → score -1 → negative
        self.assertEqual(self.a.predict_label("Today was a terrible day"), "negative")

    def test_negated_positive_classified_as_negative(self):
        # "I am not happy about this" is in dataset with true label "negative"
        self.assertEqual(self.a.predict_label("I am not happy about this"), "negative")

    def test_edge_cases_found_in_dataset(self):
        # Dataset contains posts with emojis (:)) and negation ("not happy")
        result = self.report.evaluate(SAMPLE_POSTS, TRUE_LABELS)
        self.assertGreater(len(result["edge_cases"]), 0)

    def test_classification_report_is_non_empty_string(self):
        result = self.report.evaluate(SAMPLE_POSTS, TRUE_LABELS)
        self.assertIsInstance(result["classification_report"], str)
        self.assertGreater(len(result["classification_report"]), 0)

    def test_label_order_contains_all_true_labels(self):
        result = self.report.evaluate(SAMPLE_POSTS, TRUE_LABELS)
        for lbl in set(TRUE_LABELS):
            self.assertIn(lbl, result["label_order"])


# ---------------------------------------------------------------------------
# ML model comparison
# ---------------------------------------------------------------------------

class TestMLModelComparison(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.a = MoodAnalyzer()
        cls.vectorizer, cls.ml_model = train_ml_model(SAMPLE_POSTS, TRUE_LABELS)
        cls.comparator = ModelComparator(cls.a, cls.vectorizer, cls.ml_model)
        cls.valid = {"positive", "negative", "neutral", "mixed"}

    def test_ml_single_prediction_is_valid_label(self):
        pred = predict_single_text("I am happy today", self.vectorizer, self.ml_model)
        self.assertIn(pred, self.valid)

    def test_compare_returns_required_keys(self):
        result = self.comparator.compare("I am happy today")
        for key in ("rule_based", "ml_model", "agree"):
            self.assertIn(key, result)

    def test_compare_agree_is_bool(self):
        result = self.comparator.compare("I am happy today")
        self.assertIsInstance(result["agree"], bool)

    def test_compare_both_labels_valid(self):
        result = self.comparator.compare("I love a terrible situation")
        self.assertIn(result["rule_based"], self.valid)
        self.assertIn(result["ml_model"], self.valid)

    def test_dataset_comparison_returns_required_keys(self):
        result = self.comparator.compare_on_dataset(SAMPLE_POSTS, TRUE_LABELS)
        for key in ("rule_accuracy", "ml_accuracy", "agreement_rate"):
            self.assertIn(key, result)

    def test_agreement_rate_in_valid_range(self):
        result = self.comparator.compare_on_dataset(SAMPLE_POSTS, TRUE_LABELS)
        self.assertGreaterEqual(result["agreement_rate"], 0.0)
        self.assertLessEqual(result["agreement_rate"], 1.0)

    def test_ml_accuracy_above_zero(self):
        result = self.comparator.compare_on_dataset(SAMPLE_POSTS, TRUE_LABELS)
        self.assertGreater(result["ml_accuracy"], 0.0)

    def test_rule_accuracy_above_zero(self):
        result = self.comparator.compare_on_dataset(SAMPLE_POSTS, TRUE_LABELS)
        self.assertGreater(result["rule_accuracy"], 0.0)

    def test_both_models_agree_on_clear_positive(self):
        # "I love this class so much" is in training data labeled positive
        result = self.comparator.compare("I love this class so much")
        self.assertEqual(result["rule_based"], "positive")
        self.assertEqual(result["ml_model"], "positive")

    def test_both_models_agree_on_clear_negative(self):
        # "Today was a terrible day" is in training data labeled negative
        result = self.comparator.compare("Today was a terrible day")
        self.assertEqual(result["rule_based"], "negative")
        self.assertEqual(result["ml_model"], "negative")

    def test_predictions_list_lengths_match(self):
        result = self.comparator.compare_on_dataset(SAMPLE_POSTS, TRUE_LABELS)
        self.assertEqual(len(result["rule_predictions"]), len(SAMPLE_POSTS))
        self.assertEqual(len(result["ml_predictions"]), len(SAMPLE_POSTS))


# ---------------------------------------------------------------------------
# confidence_label() utility
# ---------------------------------------------------------------------------

class TestConfidenceLabel(unittest.TestCase):

    def test_score_zero_is_none(self):
        self.assertEqual(confidence_label(0), "NONE")

    def test_score_pos_one_is_low(self):
        self.assertEqual(confidence_label(1), "LOW")

    def test_score_neg_one_is_low(self):
        self.assertEqual(confidence_label(-1), "LOW")

    def test_score_pos_two_is_medium(self):
        self.assertEqual(confidence_label(2), "MEDIUM")

    def test_score_neg_two_is_medium(self):
        self.assertEqual(confidence_label(-2), "MEDIUM")

    def test_score_three_is_high(self):
        self.assertEqual(confidence_label(3), "HIGH")

    def test_large_score_is_high(self):
        self.assertEqual(confidence_label(10), "HIGH")

    def test_returns_string(self):
        self.assertIsInstance(confidence_label(5), str)


# ---------------------------------------------------------------------------
# ReliabilityReport
# ---------------------------------------------------------------------------

class TestReliabilityReport(unittest.TestCase):

    def setUp(self):
        self.a = MoodAnalyzer()
        self.r = ReliabilityReport(self.a)

    def test_evaluate_returns_dict(self):
        self.assertIsInstance(self.r.evaluate(SAMPLE_POSTS, TRUE_LABELS), dict)

    def test_accuracy_in_valid_range(self):
        acc = self.r.evaluate(SAMPLE_POSTS, TRUE_LABELS)["accuracy"]
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_perfect_accuracy_on_custom_data(self):
        custom = MoodAnalyzer(positive_words=["fantastic"], negative_words=["dreadful"])
        report = ReliabilityReport(custom)
        result = report.evaluate(["fantastic day", "dreadful day"], ["positive", "negative"])
        self.assertEqual(result["accuracy"], 1.0)

    def test_confusion_matrix_present(self):
        result = self.r.evaluate(SAMPLE_POSTS, TRUE_LABELS)
        self.assertIn("confusion_matrix", result)

    def test_edge_cases_list_present(self):
        result = self.r.evaluate(SAMPLE_POSTS, TRUE_LABELS)
        self.assertIn("edge_cases", result)
        self.assertIsInstance(result["edge_cases"], list)

    def test_predictions_in_result(self):
        result = self.r.evaluate(SAMPLE_POSTS, TRUE_LABELS)
        self.assertIn("predictions", result)

    def test_single_example_works(self):
        result = self.r.evaluate(["I love this"], ["positive"])
        self.assertGreaterEqual(result["accuracy"], 0.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
