from __future__ import annotations

import runpy
import unittest
from typing import Any


class QuickstartTests(unittest.TestCase):
    def test_60_second_quickstart_runs_with_core_api(self) -> None:
        namespace = runpy.run_path("examples/quickstart_60_second.py")
        summary = namespace["run_quickstart"]()

        self.assertIsInstance(summary, dict)
        typed_summary: dict[str, Any] = summary
        self.assertGreater(typed_summary["csv_to_kg"]["property_nodes"], 0)
        self.assertGreater(typed_summary["text_to_kg"]["entities"], 0)
        self.assertGreater(typed_summary["json_to_kg"]["property_nodes"], 0)
        self.assertIn(
            typed_summary["best_benchmark_technique"],
            {"regex_entity_extractor", "hybrid_rules_gazetteer_regex"},
        )
        self.assertGreater(typed_summary["export_preview"]["rdf_turtle_lines"], 0)
        self.assertGreater(typed_summary["export_preview"]["neo4j_cypher_statements"], 0)
        self.assertGreater(typed_summary["export_preview"]["networkx_nodes"], 0)


if __name__ == "__main__":
    unittest.main()
