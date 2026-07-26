from __future__ import annotations

import runpy
import unittest
from typing import Any

from kg_construction import load_sources
from kg_construction.construction.workflow import DraftConstructionWorkflow
from kg_construction.research.workflow import (
    draft_workflow_for_source_type,
)
from kg_construction.unstructured.pipeline import UnstructuredPipeline


class QuickstartTests(unittest.TestCase):
    def test_complete_quickstart_runs_with_core_api(self) -> None:
        namespace = runpy.run_path("examples/quickstart.py")

        def draft_construct(input_data, *, source_type=None, **_kwargs):
            sources = load_sources(input_data, source_type=source_type)
            if source_type == "text":
                workflow = DraftConstructionWorkflow(
                    extractors=[UnstructuredPipeline()]
                )
            else:
                workflow = draft_workflow_for_source_type(source_type)
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
        self.assertFalse(typed_summary["csv_to_kg"]["release_eligible"])
        self.assertFalse(typed_summary["text_to_kg"]["release_eligible"])
        self.assertGreater(typed_summary["export_preview"]["rdf_turtle_lines"], 0)
        self.assertGreater(typed_summary["export_preview"]["neo4j_cypher_statements"], 0)
        self.assertGreater(typed_summary["export_preview"]["networkx_nodes"], 0)


if __name__ == "__main__":
    unittest.main()
