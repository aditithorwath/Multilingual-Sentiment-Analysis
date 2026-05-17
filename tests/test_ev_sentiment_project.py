import csv
import tempfile
import unittest
from pathlib import Path

from ev_sentiment_project import (
    SENTIMENT_LABELS,
    analyze_rows,
    load_dataset,
    write_language_summary,
)


class EvSentimentProjectTest(unittest.TestCase):
    def test_load_dataset_rejects_missing_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.csv"
            path.write_text("text,language\nsample,en\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Missing required columns"):
                load_dataset(path)

    def test_analyze_rows_counts_sentiments_and_languages(self):
        rows = [
            {"text": "Battery issue", "language": "en", "sentiment": "0", "sentiment_label": "Negative"},
            {"text": "Battery safety", "language": "en", "sentiment": "1", "sentiment_label": "Neutral"},
            {"text": "La seguridad", "language": "es", "sentiment": "0", "sentiment_label": "Negative"},
        ]

        analysis = analyze_rows(rows)

        self.assertEqual(analysis.total_rows, 3)
        self.assertEqual(analysis.languages, ["en", "es"])
        self.assertEqual(analysis.sentiment_counts["Negative"], 2)
        self.assertEqual(analysis.sentiment_counts["Neutral"], 1)
        self.assertEqual(analysis.language_sentiment_counts["en"]["Negative"], 1)
        self.assertEqual(analysis.language_sentiment_counts["es"]["Negative"], 1)

    def test_write_language_summary_creates_language_rows_with_required_columns(self):
        rows = [
            {"text": "Battery issue", "language": "en", "sentiment": "0", "sentiment_label": "Negative"},
            {"text": "Battery safety", "language": "en", "sentiment": "1", "sentiment_label": "Neutral"},
            {"text": "Le rappel", "language": "fr", "sentiment": "2", "sentiment_label": "Positive"},
        ]
        analysis = analyze_rows(rows)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "summary.csv"
            write_language_summary(analysis, output_path)

            with output_path.open("r", encoding="utf-8", newline="") as file:
                summary_rows = list(csv.DictReader(file))

        self.assertEqual(
            list(summary_rows[0].keys()),
            ["language", *SENTIMENT_LABELS, "total"],
        )
        self.assertEqual(summary_rows[0]["language"], "en")
        self.assertEqual(summary_rows[0]["Negative"], "1")
        self.assertEqual(summary_rows[0]["Neutral"], "1")
        self.assertEqual(summary_rows[0]["total"], "2")


if __name__ == "__main__":
    unittest.main()
