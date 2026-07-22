from __future__ import annotations

import unittest
from pathlib import Path

from kg_construction import (
    ConstructionResult,
    ConstructionWorkflow,
    DataToKGConfig,
    SourceRecord,
    TechniqueEvaluationReport,
    construct_kg,
    default_registry,
    evaluate_techniques,
    load_sources,
    profile_dataset,
)


class PublicAPITests(unittest.TestCase):
    def test_top_level_exports_include_framework_facade(self) -> None:
        registry = default_registry()

        self.assertEqual(DataToKGConfig().pipeline.name, "complete_data_to_kg")
        self.assertTrue(registry.get("csv_loader"))
        self.assertTrue(ConstructionWorkflow)
        self.assertTrue(ConstructionResult)
        self.assertTrue(TechniqueEvaluationReport)

    def test_load_sources_and_profile_dataset_for_csv(self) -> None:
        sources = load_sources("examples/data_to_kg/sample.csv")
        profile = profile_dataset(sources)

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].source_type, "csv")
        self.assertEqual(profile.data_type, "structured")
        self.assertIn("id", profile.primary_keys)

    def test_construct_kg_from_csv_path(self) -> None:
        result = construct_kg("examples/data_to_kg/sample.csv")

        self.assertIsInstance(result, ConstructionResult)
        self.assertGreaterEqual(len(result.batch.entities), 3)
        self.assertGreaterEqual(len(result.property_graph.nodes), 3)
        self.assertTrue(result.validation_report.valid)

    def test_construct_kg_from_raw_text(self) -> None:
        result = construct_kg(
            "Contact asha@example.com on 2026-07-09. INCIDENT-1042 affected Recommendation API.",
            source_type="text",
            source_name="memory://public-api-test",
        )

        self.assertEqual(result.batch.source_records[0].source_name, "memory://public-api-test")
        self.assertTrue(any(entity.name == "2026-07-09" for entity in result.batch.entities))
        self.assertTrue(any(entity.name == "INCIDENT-1042" for entity in result.batch.entities))

    def test_construct_kg_accepts_preloaded_sources(self) -> None:
        source = SourceRecord.from_text(
            "INCIDENT-1042 affected Recommendation API.",
            source_name="memory://preloaded",
        )
        result = construct_kg([source], source_type="text")

        self.assertEqual(result.batch.source_records[0].source_name, "memory://preloaded")
        self.assertGreater(len(result.batch.entities), 0)

    def test_construct_kg_routes_each_source_in_a_mixed_directory(self) -> None:
        result = construct_kg("tests/fixtures/golden_benchmarks/mixed")

        self.assertEqual(len(result.batch.source_records), 3)
        self.assertEqual(
            {source.source_type for source in result.batch.source_records},
            {"csv", "json", "text"},
        )
        self.assertIn("Record", {entity.entity_type for entity in result.batch.entities})
        self.assertIn("JSONRecord", {entity.entity_type for entity in result.batch.entities})
        self.assertIn("Identifier", {entity.entity_type for entity in result.batch.entities})

    def test_default_structured_identity_is_source_scoped_after_redaction(self) -> None:
        sources = [
            SourceRecord.from_rows(
                [{"email": "one@example.com", "department": "Research"}],
                source_type="csv",
                source_name="memory://source-a",
            ),
            SourceRecord.from_rows(
                [{"email": "two@example.com", "department": "Research"}],
                source_type="csv",
                source_name="memory://source-b",
            ),
        ]

        result = construct_kg(sources)
        canonical_ids = {
            str(entity.properties["canonical_entity_id"])
            for entity in result.batch.entities
        }

        self.assertEqual(len(result.batch.entities), 2)
        self.assertEqual(len(canonical_ids), 2)
        self.assertEqual(len(result.property_graph.nodes), 2)
        self.assertTrue(all(entity.name != "[REDACTED]" for entity in result.batch.entities))

    def test_evaluate_techniques_from_path(self) -> None:
        report = evaluate_techniques(
            Path("examples/data_to_kg/sample.txt"),
            source_type="text",
            techniques="regex_entity_extractor,hybrid_rules_gazetteer_regex",
        )

        self.assertIsInstance(report, TechniqueEvaluationReport)
        self.assertEqual(
            {candidate.technique_name for candidate in report.candidates},
            {"regex_entity_extractor", "hybrid_rules_gazetteer_regex"},
        )
        self.assertIsNotNone(report.best_technique)


if __name__ == "__main__":
    unittest.main()
