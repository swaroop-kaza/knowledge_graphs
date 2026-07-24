from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from kg_construction.application.config import DataToKGConfig
from kg_construction.application.production import build_production_services
from kg_construction.application.registry import default_registry
from kg_construction.application.workflow import ConstructionWorkflow
from kg_construction.domain.claim import DocumentRef
from kg_construction.domain.models import ConstructionBatch, SourceRecord
from kg_construction.infrastructure.extractors.regex import RegexEntityExtractor
from kg_construction.infrastructure.graph_builders.evidence_graph import (
    EvidenceGraphBuilder,
)
from kg_construction.infrastructure.mapping.direct_table_mapper import (
    DirectTableMapper,
    DirectTableMappingConfig,
)
from kg_construction.infrastructure.ontology.validation import (
    EvidenceRequiredValidator,
    ValidationFinding,
    ValidationReport,
)
from kg_construction.infrastructure.production import (
    ErrorQuarantine,
    HumanReviewQueue,
    IncrementalSyncPolicy,
    PIIRedactionPolicy,
    RunManifestWriter,
    StructuredLoggingConfig,
    WatermarkStore,
    configure_structured_logging,
)
from kg_construction.infrastructure.resolution.exact import ExactKeyEntityResolver


class FailingExtractor:
    name = "failing_extractor"

    def extract(self, source: SourceRecord) -> ConstructionBatch:
        _ = source
        raise RuntimeError("simulated extractor failure for quarantine")


class ProductionHardeningTests(unittest.TestCase):
    def test_pii_policy_redacts_text_records_metadata_and_structured_logs(self) -> None:
        policy = PIIRedactionPolicy()
        source = SourceRecord.from_text(
            "Contact asha@example.com from 10.10.1.8 with api_key=secret123",
            source_name="memory://pii",
            metadata={"owner": "asha@example.com"},
        )
        source.records.append({"email": "asha@example.com", "phone": "415-555-0101"})
        redacted = policy.redact_source_record(source)

        self.assertNotIn("asha@example.com", redacted.content or "")
        self.assertEqual(redacted.records[0]["email"], "[REDACTED]")
        self.assertEqual(redacted.records[0]["phone"], "[REDACTED]")
        self.assertEqual(redacted.metadata["owner"], "[REDACTED]")
        self.assertTrue(redacted.metadata["_pii_redaction"]["applied"])
        self.assertNotEqual(redacted.source_id, source.source_id)
        self.assertEqual(redacted.document_version, redacted.source_id)
        self.assertEqual(DocumentRef.from_source(redacted).version, redacted.source_id)
        lineage = redacted.metadata["_source_lineage"]
        self.assertEqual(lineage["relation"], "DERIVED_FROM")
        self.assertEqual(lineage["parent_source_id"], source.source_id)
        self.assertEqual(lineage["parent_document_version"], source.source_id)
        self.assertEqual(lineage["root_source_id"], source.source_id)
        self.assertEqual(lineage["derived_source_id"], redacted.source_id)

        repeated = policy.redact_source_record(source)
        self.assertEqual(repeated.source_id, redacted.source_id)
        self.assertEqual(repeated.metadata["_source_lineage"], lineage)
        self.assertIn("asha@example.com", source.content or "")

        evidence_graph = EvidenceGraphBuilder().build(
            ConstructionBatch(source_records=[redacted])
        )
        derived_from = [
            edge for edge in evidence_graph.edges if edge.type == "DERIVED_FROM"
        ]
        self.assertEqual(len(derived_from), 1)
        self.assertEqual(derived_from[0].source_id, redacted.source_id)
        self.assertEqual(derived_from[0].target_id, source.source_id)
        self.assertIn(source.source_id, {node.node_id for node in evidence_graph.nodes})

        stream = io.StringIO()
        logger = configure_structured_logging(
            StructuredLoggingConfig(enabled=True, redact_pii=True),
            stream=stream,
            pii_policy=policy,
        )
        logger.info("pii_test", contact="asha@example.com")
        payload = json.loads(stream.getvalue())

        self.assertEqual(payload["event"], "pii_test")
        self.assertEqual(payload["contact"], "[REDACTED]")

    def test_pii_policy_preserves_identity_when_source_payload_does_not_change(
        self,
    ) -> None:
        source = SourceRecord.from_text(
            "No sensitive values are present.",
            source_name="memory://pii/no-change",
        )

        redacted = PIIRedactionPolicy().redact_source_record(source)

        self.assertEqual(redacted.source_id, source.source_id)
        self.assertFalse(redacted.metadata["_pii_redaction"]["snapshot_mutated"])
        self.assertNotIn("_source_lineage", redacted.metadata)

    def test_redacted_snapshots_that_collapse_to_same_text_keep_distinct_lineage(
        self,
    ) -> None:
        first = SourceRecord.from_text(
            "Contact first@example.com",
            source_name="memory://pii/collision",
        )
        second = SourceRecord.from_text(
            "Contact second@example.com",
            source_name="memory://pii/collision",
        )
        policy = PIIRedactionPolicy()

        redacted_first = policy.redact_source_record(first)
        redacted_second = policy.redact_source_record(second)

        self.assertEqual(redacted_first.content, redacted_second.content)
        self.assertNotEqual(redacted_first.source_id, redacted_second.source_id)
        self.assertEqual(
            redacted_first.metadata["_source_lineage"]["parent_source_id"],
            first.source_id,
        )
        self.assertEqual(
            redacted_second.metadata["_source_lineage"]["parent_source_id"],
            second.source_id,
        )

    def test_redacted_configured_identity_columns_do_not_false_merge(self) -> None:
        source = SourceRecord.from_rows(
            [
                {"email": "one@example.com", "department": "Research"},
                {"email": "two@example.com", "department": "Research"},
            ],
            source_type="csv",
            source_name="memory://pii-identities",
        )
        redacted = PIIRedactionPolicy().redact_source_record(source)
        batch = DirectTableMapper(
            DirectTableMappingConfig(
                node_type="Person",
                id_columns=["email"],
            )
        ).map_source(redacted)

        self.assertEqual(len(batch.entities), 2)
        self.assertEqual(
            len({entity.properties["identity_key"] for entity in batch.entities}),
            2,
        )

    def test_missing_configured_identity_values_do_not_false_merge(self) -> None:
        source = SourceRecord.from_rows(
            [
                {"external_id": None, "name": "First unknown"},
                {"external_id": "", "name": "Second unknown"},
            ],
            source_type="csv",
            source_name="unknown-identities.csv",
        )
        mapped = DirectTableMapper(
            DirectTableMappingConfig(
                node_type="Person",
                id_columns=["external_id"],
                name_column="name",
            )
        ).map_source(source)
        resolved = ExactKeyEntityResolver().resolve(mapped)

        self.assertEqual(
            len(
                {
                    entity.properties["canonical_entity_id"]
                    for entity in resolved.entities
                }
            ),
            2,
        )
        self.assertEqual(
            [entity.name for entity in mapped.entities],
            ["First unknown", "Second unknown"],
        )

    def test_row_fallback_identity_is_stable_when_source_rows_reorder(self) -> None:
        rows = [
            {"name": "Ada", "department": "Research"},
            {"name": "Grace", "department": "Platform"},
        ]
        mapper = DirectTableMapper(DirectTableMappingConfig(node_type="Person"))
        forward = ExactKeyEntityResolver().resolve(
            mapper.map_source(
                SourceRecord.from_rows(
                    rows,
                    source_type="csv",
                    source_name="people.csv",
                )
            )
        )
        reversed_batch = ExactKeyEntityResolver().resolve(
            mapper.map_source(
                SourceRecord.from_rows(
                    list(reversed(rows)),
                    source_type="csv",
                    source_name="people.csv",
                )
            )
        )

        forward_ids = {
            entity.name: entity.properties["canonical_entity_id"]
            for entity in forward.entities
        }
        reversed_ids = {
            entity.name: entity.properties["canonical_entity_id"]
            for entity in reversed_batch.entities
        }
        self.assertEqual(forward_ids, reversed_ids)

    def test_watermark_store_filters_and_commits_incremental_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = WatermarkStore(Path(tmpdir) / "watermarks.json")
            policy = IncrementalSyncPolicy()
            old_source = SourceRecord.from_text(
                "old",
                source_name="memory://sync",
                metadata={"updated_at": "2026-07-09T10:00:00Z"},
            )
            new_source = SourceRecord.from_text(
                "new",
                source_name="memory://sync",
                metadata={"updated_at": "2026-07-09T11:00:00Z"},
            )

            first = policy.apply([old_source], store)
            policy.commit(first.eligible_sources, store)
            second = policy.apply([old_source], store)
            third = policy.apply([new_source], store)

        self.assertEqual(len(first.eligible_sources), 1)
        self.assertEqual(len(second.skipped_sources), 1)
        self.assertEqual(len(third.eligible_sources), 1)

    def test_content_hash_fallback_detects_change_without_ordering_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = WatermarkStore(Path(tmpdir) / "watermarks.json")
            policy = IncrementalSyncPolicy()
            first_source = SourceRecord.from_text(
                "snapshot one",
                source_name="memory://hash-sync",
            )
            changed_source = SourceRecord.from_text(
                "snapshot two",
                source_name="memory://hash-sync",
            )

            first = policy.apply([first_source], store)
            policy.commit(first.eligible_sources, store)
            replay = policy.apply([first_source], store)
            changed = policy.apply([changed_source], store)

        self.assertEqual(len(replay.skipped_sources), 1)
        self.assertEqual(len(changed.eligible_sources), 1)

    def test_numeric_watermarks_use_numeric_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = WatermarkStore(Path(tmpdir) / "numeric-watermarks.json")
            policy = IncrementalSyncPolicy(watermark_field="sequence")
            ninth = SourceRecord.from_text(
                "ninth",
                source_name="memory://numeric-watermark",
                metadata={"sequence": "9"},
            )
            tenth = SourceRecord.from_text(
                "tenth",
                source_name="memory://numeric-watermark",
                metadata={"sequence": "10"},
            )
            policy.commit([ninth], store)

            result = policy.apply([tenth], store)

        self.assertEqual(len(result.eligible_sources), 1)

    def test_quarantine_and_human_review_queue_persist_validation_findings(self) -> None:
        finding = ValidationFinding(
            object_id="entity_1",
            object_type="entity",
            rule="confidence_threshold",
            message="Candidate needs review.",
            policy="quarantine",
        )
        report = ValidationReport(findings=[finding], checked_count=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            quarantine = ErrorQuarantine(Path(tmpdir) / "quarantine.jsonl")
            queue = HumanReviewQueue(Path(tmpdir) / "review.json")
            records = quarantine.quarantine_findings(
                batch=ConstructionBatch(),
                validation_report=report,
            )
            items = queue.enqueue_batch_review(batch=ConstructionBatch(), findings=[finding])
            approved = queue.decide(items[0].review_id, status="approved", reviewer="unit-test")

            loaded_records = quarantine.load_records()
            reloaded_queue = HumanReviewQueue(Path(tmpdir) / "review.json")

        self.assertEqual(len(records), 1)
        self.assertEqual(loaded_records[0].rule, "confidence_threshold")
        self.assertEqual(items[0].status, "pending_review")
        self.assertEqual(approved.status, "approved")
        self.assertEqual(len(reloaded_queue.pending()), 0)

    def test_workflow_writes_manifest_and_quarantines_component_errors(self) -> None:
        source = SourceRecord.from_text(
            "Contact asha@example.com on 2026-07-09. INCIDENT-1042 affected Recommendation API.",
            source_name="memory://workflow",
            metadata={"updated_at": "2026-07-09T10:00:00Z"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            workflow = ConstructionWorkflow(
                extractors=[FailingExtractor(), RegexEntityExtractor()],
                validators=[EvidenceRequiredValidator(policy="quarantine")],
                manifest_writer=RunManifestWriter(Path(tmpdir) / "manifests"),
                quarantine=ErrorQuarantine(Path(tmpdir) / "quarantine.jsonl"),
                human_review_queue=HumanReviewQueue(Path(tmpdir) / "review.json"),
                pii_policy=PIIRedactionPolicy(),
                continue_on_error=True,
                pipeline_name="prod_test",
                config_payload={"pipeline": "prod_test"},
            )
            result = workflow.run([source])
            manifest_files = list((Path(tmpdir) / "manifests").glob("*.manifest.json"))
            quarantine_records = ErrorQuarantine(Path(tmpdir) / "quarantine.jsonl").load_records()

        self.assertIsNotNone(result.run_manifest)
        self.assertEqual(result.run_manifest.status if result.run_manifest else "", "completed")
        self.assertTrue(manifest_files)
        self.assertGreater(len(result.batch.entities), 0)
        self.assertTrue(any(record.reason == "exception" for record in quarantine_records))
        self.assertNotIn("asha@example.com", result.batch.source_records[0].content or "")

    def test_production_services_are_config_driven(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DataToKGConfig.model_validate(
                {
                    "production": {
                        "structured_logging": {"enabled": False},
                        "run_manifest": {
                            "enabled": True,
                            "output_dir": str(Path(tmpdir) / "manifests"),
                        },
                        "incremental_sync": {
                            "enabled": True,
                            "watermark_path": str(Path(tmpdir) / "watermarks.json"),
                        },
                        "error_quarantine": {
                            "enabled": True,
                            "path": str(Path(tmpdir) / "quarantine.jsonl"),
                        },
                        "human_review_queue": {
                            "enabled": True,
                            "path": str(Path(tmpdir) / "review.json"),
                        },
                        "continue_on_error": True,
                    }
                }
            )
            services = build_production_services(config)

        self.assertIsNotNone(services.manifest_writer)
        self.assertIsNotNone(services.watermark_store)
        self.assertIsNotNone(services.quarantine)
        self.assertIsNotNone(services.human_review_queue)
        self.assertTrue(services.continue_on_error)

    def test_config_validation_catches_bad_values_and_unknown_techniques(self) -> None:
        self.assertEqual(DataToKGConfig.from_yaml("configs/data_to_kg_complete.example.yaml").pipeline.name, "complete_data_to_kg")
        with self.assertRaises(ValidationError):
            DataToKGConfig.model_validate({"llm": {"temperature": 3.5}})
        with self.assertRaises(ValidationError):
            DataToKGConfig.model_validate({"loaders": {"spreadsheet_magic": {"enabled": True}}})
        with self.assertRaises(ValidationError):
            DataToKGConfig.model_validate(
                {
                    "production": {
                        "run_manifest": {"enabled": True, "output_dir": ""}
                    }
                }
            )

    def test_registry_includes_production_hardening_capabilities(self) -> None:
        registry = default_registry()

        self.assertIn("production", registry.categories())
        for name in {
            "error_quarantine",
            "incremental_sync_policy",
            "pii_redaction_policy",
            "run_manifest_writer",
            "structured_logging",
            "watermark_store",
        }:
            self.assertEqual(registry.get(name).status, "implemented")


if __name__ == "__main__":
    unittest.main()
