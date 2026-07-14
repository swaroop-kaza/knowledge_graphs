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
from kg_construction.domain.models import ConstructionBatch, SourceRecord
from kg_construction.infrastructure.extractors.regex import RegexEntityExtractor
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
