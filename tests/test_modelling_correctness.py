"""Modelling-correctness gold fixtures (CI-verified on every change)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from kg_construction.evaluation.modelling_correctness import (
    evaluate_structured_case,
    evaluate_unstructured_case,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "gold_standards"


def _load(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


class StructuredGoldGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = _load("modelling_correctness_structured.json")
        self.result = evaluate_structured_case(
            case_id=self.fixture["case_id"],
            records=self.fixture["records"],
            table_name=self.fixture["table_name"],
            record_type=self.fixture["record_type"],
            gold_graph=self.fixture["gold_graph"],
        )

    def test_output_graph_conforms_to_gold(self) -> None:
        failed = [name for name, passed in self.result.checks.items() if not passed]
        self.assertEqual(failed, [], f"failed checks: {failed}")
        self.assertTrue(self.result.passed)

    def test_categorical_dedup_and_fk_edges(self) -> None:
        # The two headline guarantees, called out explicitly for CI visibility.
        self.assertEqual(self.result.metrics["categorical_duplication_rate"], 0.0)
        self.assertEqual(self.result.metrics["nodes:Department"], 3.0)  # deduplicated dimension
        self.assertEqual(self.result.metrics["edges:has_department"], 8.0)
        self.assertEqual(self.result.metrics["edges:manager"], 7.0)  # foreign-key edges


class UnstructuredCoverageTests(unittest.TestCase):
    def test_entity_coverage_meets_minimum(self) -> None:
        fixture = _load("modelling_correctness_unstructured.json")
        result = evaluate_unstructured_case(
            case_id=fixture["case_id"],
            text=fixture["text"],
            gold_entities=fixture["gold_entities"],
            minimum_entity_coverage=fixture["minimum_entity_coverage"],
        )
        self.assertTrue(result.passed)
        self.assertGreaterEqual(result.metrics["entity_coverage"], fixture["minimum_entity_coverage"])
        # Regex captures the 4 structured entities of the 5 gold salient mentions.
        self.assertEqual(result.metrics["entity_coverage"], 0.8)


if __name__ == "__main__":
    unittest.main()
