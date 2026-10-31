"""Tests for the strict parser and the metrics in evaluate.py.

Standard library only. Run with: python test_evaluate.py
"""

import json
import unittest
from pathlib import Path

from evaluate import INVALID, SAFE, compute_metrics, parse_output

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_FILES = [DATA_DIR / f"{split}.jsonl" for split in ["train", "valid", "test"]]


def gold_labels(path):
    return [json.loads(line)["messages"][2]["content"] for line in path.read_text(encoding="utf-8").splitlines()]


class ParserTests(unittest.TestCase):
    def test_accepts_every_gold_label_in_the_data(self):
        labels = [label for path in DATA_FILES for label in gold_labels(path)]
        self.assertEqual(len(labels), 2998)
        for label in sorted(set(labels)):
            self.assertEqual(parse_output(label), label)
        self.assertEqual(sum(1 for label in labels if parse_output(label) == INVALID), 0)

    def test_strips_only_surrounding_whitespace(self):
        self.assertEqual(parse_output("  SAFE\n"), "SAFE")
        self.assertEqual(parse_output("\nLEAKED: jwt  "), "LEAKED: jwt")

    def test_rejects_near_misses(self):
        for raw in ["Safe", "SAFE.", "LEAKED: password.", "LEAKED: api_key", "SAFE\nThe snippet is safe",
                    "leaked: jwt", "LEAKED:password", "LEAKED:  password", "LEAKED", "", "SAFE SAFE"]:
            with self.subTest(raw=raw):
                self.assertEqual(parse_output(raw), INVALID)


class MetricTests(unittest.TestCase):
    def test_gold_labels_as_predictions_score_perfectly(self):
        golds = gold_labels(DATA_DIR / "test.jsonl")
        m = compute_metrics(golds, golds)
        self.assertEqual(m["f1"], 1.0)
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["binary_accuracy"], 1.0)
        self.assertEqual(m["type_accuracy"], 1.0)
        self.assertEqual(m["invalid"], 0)

    def test_hand_made_case_with_known_counts(self):
        golds = [
            "LEAKED: password",        # predicted correctly          -> TP, exact
            "LEAKED: jwt",             # flagged with the wrong type  -> TP, wrong type
            "LEAKED: aws_access_key",  # missed                       -> FN
            "LEAKED: stripe_key",      # INVALID output               -> FN
            SAFE,                      # correct                      -> TN
            SAFE,                      # flagged                      -> FP
            SAFE,                      # INVALID output               -> wrong, but not a FP
            SAFE,                      # correct                      -> TN
        ]
        preds = [
            "LEAKED: password", "LEAKED: password", SAFE, INVALID,
            SAFE, "LEAKED: jwt", INVALID, SAFE,
        ]
        m = compute_metrics(golds, preds)
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"], m["invalid"]), (2, 1, 2, 2, 2))
        self.assertAlmostEqual(m["precision"], 2 / 3)
        self.assertAlmostEqual(m["recall"], 2 / 4)
        self.assertAlmostEqual(m["f1"], 4 / 7)  # 2 * (2/3) * (1/2) / (2/3 + 1/2)
        self.assertAlmostEqual(m["accuracy"], 3 / 8)
        self.assertAlmostEqual(m["binary_accuracy"], 4 / 8)
        self.assertAlmostEqual(m["type_accuracy"], 1 / 2)
        self.assertAlmostEqual(m["format_error_rate"], 2 / 8)

    def test_undefined_rates_are_none(self):
        m = compute_metrics([SAFE, SAFE], [SAFE, SAFE])  # no positives at all
        self.assertIsNone(m["precision"])
        self.assertIsNone(m["recall"])
        self.assertIsNone(m["f1"])
        self.assertEqual(m["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
