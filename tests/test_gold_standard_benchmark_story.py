from __future__ import annotations

import csv
import unittest
from pathlib import Path

from scripts.generate_gold_standard_benchmark_story import (
    STORY_CSV,
    STORY_JSON,
    STORY_MD,
    generate_gold_standard_benchmark_story,
)


class GoldStandardBenchmarkStoryTests(unittest.TestCase):
    def test_committed_story_has_all_gold_benchmark_types(self) -> None:
        self.assertTrue(STORY_CSV.exists())

        with STORY_CSV.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 5)
        self.assertEqual(
            {row["benchmark_type"] for row in rows},
            {
                "NER span benchmark",
                "Relation extraction benchmark",
                "Triple benchmark",
                "Entity resolution benchmark",
                "GraphRAG/retrieval benchmark",
            },
        )
        self.assertTrue(all(float(row["f1"]) >= 0.0 for row in rows))

    def test_story_generation_is_reproducible_for_core_fields(self) -> None:
        generated, _details = generate_gold_standard_benchmark_story()
        with STORY_CSV.open(encoding="utf-8") as handle:
            committed = list(csv.DictReader(handle))

        generated_key_rows = [
            (
                row["benchmark_type"],
                row["fixture_path"],
                row["technique"],
                f"{row['precision']}",
                f"{row['recall']}",
                f"{row['f1']}",
                f"{row['support']}",
            )
            for row in generated
        ]
        committed_key_rows = [
            (
                row["benchmark_type"],
                row["fixture_path"],
                row["technique"],
                row["precision"],
                row["recall"],
                row["f1"],
                row["support"],
            )
            for row in committed
        ]
        self.assertEqual(generated_key_rows, committed_key_rows)

    def test_markdown_and_json_story_are_public_readable_and_small(self) -> None:
        self.assertTrue(STORY_MD.exists())
        self.assertTrue(STORY_JSON.exists())
        self.assertLess(Path(STORY_MD).stat().st_size, 20_000)
        markdown = STORY_MD.read_text(encoding="utf-8")
        self.assertIn("# Gold-Standard Benchmark Story", markdown)
        self.assertIn("GraphRAG/retrieval benchmark", markdown)
        self.assertIn("not a broad model leaderboard", markdown)
        self.assertIn("\"benchmark_count\": 5", STORY_JSON.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
