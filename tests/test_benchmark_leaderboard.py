from __future__ import annotations

import csv
import unittest
from pathlib import Path

from scripts.generate_benchmark_leaderboard import generate_leaderboard_rows


class BenchmarkLeaderboardTests(unittest.TestCase):
    def test_committed_leaderboard_has_required_columns_and_dataset_types(self) -> None:
        path = Path("reports/benchmark_leaderboard.csv")
        self.assertTrue(path.exists())

        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertGreaterEqual(len(rows), 10)
        self.assertEqual(
            {
                "dataset_type",
                "technique",
                "entity_coverage",
                "relation_coverage",
                "evidence_coverage",
                "validation_pass_rate",
                "latency_seconds",
                "memory_mb",
                "resource_usage",
                "final_score",
            }
            - set(rows[0]),
            set(),
        )
        self.assertEqual({row["dataset_type"] for row in rows}, {"csv", "json", "mixed", "text"})

    def test_leaderboard_generation_is_reproducible_for_core_fields(self) -> None:
        generated = generate_leaderboard_rows()
        with Path("reports/benchmark_leaderboard.csv").open(encoding="utf-8") as handle:
            committed = list(csv.DictReader(handle))

        generated_key_rows = [
            (
                str(row["rank"]),
                row["dataset_type"],
                row["technique"],
                f"{row['entity_coverage']}",
                f"{row['relation_coverage']}",
                f"{row['evidence_coverage']}",
                f"{row['validation_pass_rate']}",
                f"{row['latency_seconds']}",
                f"{row['memory_mb']}",
                f"{row['final_score']}",
            )
            for row in generated
        ]
        committed_key_rows = [
            (
                row["rank"],
                row["dataset_type"],
                row["technique"],
                row["entity_coverage"],
                row["relation_coverage"],
                row["evidence_coverage"],
                row["validation_pass_rate"],
                row["latency_seconds"],
                row["memory_mb"],
                row["final_score"],
            )
            for row in committed
        ]
        self.assertEqual(generated_key_rows, committed_key_rows)

    def test_markdown_leaderboard_is_small(self) -> None:
        path = Path("reports/benchmark_leaderboard.md")
        self.assertTrue(path.exists())
        self.assertLess(path.stat().st_size, 20_000)
        self.assertIn("Big multi-dataset experiment outputs remain generated", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
