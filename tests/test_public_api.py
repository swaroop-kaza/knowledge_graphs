from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from kg_construction import (
    ConstructionResult,
    ConstructionWorkflow,
    DataToKGConfig,
    SourceRecord,
    TechniqueEvaluationReport,
    construct_kg,
    default_registry,
    default_workflow_for_source_type,
    evaluate_techniques,
    load_sources,
    profile_dataset,
)
from kg_construction.application.unstructured_pipeline import UnstructuredPipeline
from kg_construction.infrastructure.extractors.model_backed import OfflineArtifactError


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

    def test_public_construction_rejects_an_unscored_graph(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires evaluation_suite_path"):
            construct_kg("examples/data_to_kg/sample.csv")

    def test_draft_workflow_from_csv_path(self) -> None:
        sources = load_sources("examples/data_to_kg/sample.csv")
        result = default_workflow_for_source_type("csv").run(sources)

        self.assertIsInstance(result, ConstructionResult)
        self.assertGreaterEqual(len(result.batch.entities), 3)
        self.assertGreaterEqual(len(result.property_graph.nodes), 3)
        self.assertTrue(result.validation_report.valid)
        self.assertTrue(result.quality_report.families["structured"].executed)
        self.assertTrue(result.quality_report.families["category"].executed)
        self.assertTrue(result.quality_report.families["idempotence"].executed)
        self.assertTrue(result.quality_report.families["operational"].executed)
        self.assertFalse(result.quality_report.passed)
        self.assertIn("query", result.quality_report.missing_families)

    def test_draft_workflow_from_raw_text(self) -> None:
        sources = load_sources(
            "Contact asha@example.com on 2026-07-09. INCIDENT-1042 affected Recommendation API.",
            source_type="text",
            source_name="memory://public-api-test",
        )
        result = ConstructionWorkflow(extractors=[UnstructuredPipeline()]).run(sources)

        self.assertEqual(result.batch.source_records[0].source_name, "memory://public-api-test")
        self.assertTrue(any(entity.name == "2026-07-09" for entity in result.batch.entities))
        self.assertTrue(any(entity.name == "INCIDENT-1042" for entity in result.batch.entities))

    def test_draft_workflow_accepts_preloaded_sources(self) -> None:
        source = SourceRecord.from_text(
            "INCIDENT-1042 affected Recommendation API.",
            source_name="memory://preloaded",
        )
        result = ConstructionWorkflow(extractors=[UnstructuredPipeline()]).run([source])

        self.assertEqual(result.batch.source_records[0].source_name, "memory://preloaded")
        self.assertGreater(len(result.batch.entities), 0)

    def test_draft_workflow_routes_each_source_in_a_mixed_directory(self) -> None:
        sources = load_sources("tests/fixtures/golden_benchmarks/mixed")
        with patch(
            "kg_construction.application.public_api._mandatory_unstructured_pipeline",
            return_value=UnstructuredPipeline(),
        ):
            result = default_workflow_for_source_type("mixed").run(sources)

        self.assertEqual(len(result.batch.source_records), 3)
        self.assertEqual(
            {source.source_type for source in result.batch.source_records},
            {"csv", "json", "text"},
        )
        self.assertIn("Input", {entity.entity_type for entity in result.batch.entities})
        self.assertIn("JSONRecord", {entity.entity_type for entity in result.batch.entities})
        self.assertIn("Identifier", {entity.entity_type for entity in result.batch.entities})

    def test_default_unstructured_path_requires_the_immutable_model_lock(self) -> None:
        config = DataToKGConfig.model_validate(
            {"unstructured": {"models_lock_path": "definitely-missing-models.lock.yaml"}}
        )

        with self.assertRaisesRegex(OfflineArtifactError, "model lock does not exist"):
            default_workflow_for_source_type("text", config)

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

        result = default_workflow_for_source_type("csv").run(sources)
        canonical_ids = {
            str(entity.properties["canonical_entity_id"])
            for entity in result.batch.entities
        }

        self.assertEqual(len(result.batch.entities), 2)
        self.assertEqual(len(canonical_ids), 2)
        self.assertEqual(len(result.property_graph.nodes), 2)
        self.assertTrue(all(entity.name != "[REDACTED]" for entity in result.batch.entities))

    def test_structured_declarations_drive_cross_table_public_construction(self) -> None:
        config = DataToKGConfig.model_validate(
            {
                "structured": {
                    "require_declarations": True,
                    "tables": [
                        {
                            "table_name": "teams.csv",
                            "record_type": "Team",
                            "primary_key": ["team_id"],
                        },
                        {
                            "table_name": "people.csv",
                            "record_type": "Person",
                            "primary_key": ["person_id"],
                            "foreign_keys": [
                                {
                                    "columns": ["team_id"],
                                    "target_table": "teams.csv",
                                    "target_key": ["team_id"],
                                    "relationship_name": "member_of",
                                }
                            ],
                        },
                    ],
                }
            }
        )
        sources = [
            SourceRecord.from_rows(
                [{"team_id": "T1", "name": "Research"}],
                source_type="csv",
                source_name="teams.csv",
            ),
            SourceRecord.from_rows(
                [{"person_id": "P1", "name": "Ada", "team_id": "T1"}],
                source_type="csv",
                source_name="people.csv",
            ),
        ]

        result = default_workflow_for_source_type("csv", config).run(sources)

        team_ids = {node.node_id for node in result.property_graph.nodes if "Team" in node.labels}
        membership = [
            edge for edge in result.property_graph.edges if edge.type == "member_of"
        ]
        self.assertEqual(len(team_ids), 1)
        self.assertEqual(len(membership), 1)
        self.assertIn(membership[0].target_id, team_ids)

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
