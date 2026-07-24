from __future__ import annotations

import runpy
import unittest
from typing import Any

from kg_construction import default_workflow_for_source_type, load_sources
from kg_construction.application.unstructured_pipeline import UnstructuredPipeline
from kg_construction.application.workflow import ConstructionWorkflow


class QuickstartTests(unittest.TestCase):
    def test_60_second_quickstart_runs_with_core_api(self) -> None:
        namespace = runpy.run_path("examples/quickstart_60_second.py")

        def draft_construct(input_data, *, source_type=None, **_kwargs):
            sources = load_sources(input_data, source_type=source_type)
            if source_type == "text":
                workflow = ConstructionWorkflow(extractors=[UnstructuredPipeline()])
            else:
                workflow = default_workflow_for_source_type(source_type)
            return workflow.run(sources)

        namespace["run_quickstart"].__globals__["construct_kg"] = draft_construct
        summary = namespace["run_quickstart"]()

        self.assertIsInstance(summary, dict)
        typed_summary: dict[str, Any] = summary
        self.assertGreater(typed_summary["csv_to_kg"]["property_nodes"], 0)
        self.assertGreater(typed_summary["text_to_kg"]["entities"], 0)
        self.assertIn(
            "formation",
            typed_summary["csv_to_kg"]["executed_quality_families"],
        )
        self.assertIn(
            "vector",
            typed_summary["csv_to_kg"]["missing_intended_uses"],
        )
        self.assertIn(
            typed_summary["best_benchmark_technique"],
            {"regex_entity_extractor", "hybrid_rules_gazetteer_regex"},
        )
        self.assertGreater(typed_summary["export_preview"]["rdf_turtle_lines"], 0)
        self.assertGreater(typed_summary["export_preview"]["neo4j_cypher_statements"], 0)
        self.assertGreater(typed_summary["export_preview"]["networkx_nodes"], 0)


if __name__ == "__main__":
    unittest.main()
