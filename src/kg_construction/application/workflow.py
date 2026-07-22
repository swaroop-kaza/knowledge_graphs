"""End-to-end construction workflow orchestration."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.domain.graph_objects import PropertyGraph, RDFGraph
from kg_construction.domain.models import ConstructionBatch, SourceRecord
from kg_construction.infrastructure.evaluation.metrics import ConstructionMetricsEvaluator
from kg_construction.infrastructure.evaluation.validation_metrics import (
    validation_failure_metrics,
)
from kg_construction.infrastructure.graph_builders import (
    EvidenceGraphBuilder,
    PropertyGraphBuilder,
    RDFGraphBuilder,
)
from kg_construction.infrastructure.ontology.validation import (
    CompositeValidator,
    EvidenceRequiredValidator,
    ReferentialIntegrityValidator,
    ValidationReport,
)
from kg_construction.infrastructure.production import (
    ErrorQuarantine,
    HumanReviewItem,
    HumanReviewQueue,
    IncrementalSyncPolicy,
    PIIRedactionPolicy,
    QuarantineRecord,
    RunManifest,
    RunManifestWriter,
    StructuredLogger,
    WatermarkStore,
)
from kg_construction.infrastructure.resolution.common import reconcile_entity_references


class ConstructionResult(BaseModel):
    """Complete result of one Data-to-KG workflow run."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    batch: ConstructionBatch
    validation_report: ValidationReport
    property_graph: PropertyGraph
    rdf_graph: RDFGraph
    evidence_graph: PropertyGraph
    metrics: dict[str, float] = Field(default_factory=dict)
    run_manifest: RunManifest | None = None
    quarantine_records: list[QuarantineRecord] = Field(default_factory=list)
    human_review_items: list[HumanReviewItem] = Field(default_factory=list)
    skipped_sources: list[SourceRecord] = Field(default_factory=list)


class ConstructionWorkflow:
    """Orchestrate mapping, extraction, resolution, validation, graph build, and evaluation."""

    def __init__(
        self,
        *,
        mappers: list[Any] | None = None,
        extractors: list[Any] | None = None,
        resolvers: list[Any] | None = None,
        validators: list[Any] | None = None,
        property_graph_builder: PropertyGraphBuilder | None = None,
        rdf_graph_builder: RDFGraphBuilder | None = None,
        evidence_graph_builder: EvidenceGraphBuilder | None = None,
        evaluator: ConstructionMetricsEvaluator | None = None,
        pipeline_name: str = "construction_workflow",
        config_payload: Any | None = None,
        structured_logger: StructuredLogger | None = None,
        manifest_writer: RunManifestWriter | None = None,
        watermark_store: WatermarkStore | None = None,
        incremental_sync_policy: IncrementalSyncPolicy | None = None,
        quarantine: ErrorQuarantine | None = None,
        human_review_queue: HumanReviewQueue | None = None,
        pii_policy: PIIRedactionPolicy | None = None,
        continue_on_error: bool = False,
    ) -> None:
        self.mappers = mappers or []
        self.extractors = extractors or []
        self.resolvers = resolvers or []
        configured_validators = list(validators or [EvidenceRequiredValidator()])
        if not any(
            isinstance(validator, ReferentialIntegrityValidator)
            for validator in configured_validators
        ):
            configured_validators.append(ReferentialIntegrityValidator())
        self.validator = CompositeValidator(configured_validators)
        self.property_graph_builder = property_graph_builder or PropertyGraphBuilder()
        self.rdf_graph_builder = rdf_graph_builder or RDFGraphBuilder()
        self.evidence_graph_builder = evidence_graph_builder or EvidenceGraphBuilder()
        self.evaluator = evaluator or ConstructionMetricsEvaluator()
        self.pipeline_name = pipeline_name
        self.config_payload = config_payload
        self.structured_logger = structured_logger
        self.manifest_writer = manifest_writer
        self.watermark_store = watermark_store
        self.incremental_sync_policy = incremental_sync_policy or IncrementalSyncPolicy()
        self.quarantine = quarantine
        self.human_review_queue = human_review_queue
        self.pii_policy = pii_policy
        self.continue_on_error = continue_on_error

    def run(self, sources: list[SourceRecord]) -> ConstructionResult:
        working_sources = self._redact_sources(sources)
        manifest = RunManifest.start(
            pipeline_name=self.pipeline_name,
            sources=working_sources,
            config_payload=self.config_payload,
        )
        self._write_manifest(manifest)
        self._log("info", "workflow_started", run_id=manifest.run_id, source_count=len(working_sources))
        quarantine_records: list[QuarantineRecord] = []
        human_review_items: list[HumanReviewItem] = []
        skipped_sources: list[SourceRecord] = []

        try:
            if self.watermark_store is not None:
                sync_result = self.incremental_sync_policy.apply(working_sources, self.watermark_store)
                processing_sources = sync_result.eligible_sources
                skipped_sources = sync_result.skipped_sources
                self._log(
                    "info",
                    "incremental_sync_applied",
                    eligible_sources=len(processing_sources),
                    skipped_sources=len(skipped_sources),
                )
            else:
                processing_sources = working_sources

            batch = ConstructionBatch(source_records=processing_sources)
            for source in processing_sources:
                for mapper in self.mappers:
                    batch = self._merge_component(
                        batch,
                        component=mapper,
                        method_name="map_source",
                        source=source,
                        stage="mapper",
                        quarantine_records=quarantine_records,
                    )
                for extractor in self.extractors:
                    batch = self._merge_component(
                        batch,
                        component=extractor,
                        method_name="extract",
                        source=source,
                        stage="extractor",
                        quarantine_records=quarantine_records,
                    )

            for resolver in self.resolvers:
                try:
                    batch = resolver.resolve(batch)
                except Exception as exc:
                    quarantine_records.extend(
                        self._quarantine_exception(
                            stage=f"resolver:{getattr(resolver, 'name', resolver.__class__.__name__)}",
                            error=exc,
                        )
                    )
                    self._log("exception", "resolver_failed", resolver=getattr(resolver, "name", str(resolver)))
                    if not self.continue_on_error:
                        raise

            batch = reconcile_entity_references(batch)

            validation_report = self.validator.validate(batch)
            if self.quarantine is not None:
                quarantine_records.extend(
                    self.quarantine.quarantine_findings(
                        batch=batch,
                        validation_report=validation_report,
                    )
                )
            if self.human_review_queue is not None:
                human_review_items = self.human_review_queue.enqueue_batch_review(
                    batch=batch,
                    findings=validation_report.findings,
                )

            property_graph = self.property_graph_builder.build(batch, validation_report=validation_report)
            rdf_graph = self.rdf_graph_builder.build(batch, validation_report=validation_report)
            evidence_graph = self.evidence_graph_builder.build(batch)
            metrics = self.evaluator.evaluate(batch)
            metrics.update(validation_failure_metrics(batch, validation_report))
            metrics["quarantine_record_count"] = float(len(quarantine_records))
            metrics["human_review_queue_count"] = float(len(human_review_items))
            metrics["skipped_source_count"] = float(len(skipped_sources))

            if self.watermark_store is not None:
                self.incremental_sync_policy.commit(processing_sources, self.watermark_store)

            manifest = manifest.complete(
                batch=batch,
                validation_report=validation_report,
                metadata={
                    "quarantine_records": len(quarantine_records),
                    "human_review_items": len(human_review_items),
                    "skipped_sources": len(skipped_sources),
                },
            )
            self._write_manifest(manifest)
            self._log(
                "info",
                "workflow_completed",
                run_id=manifest.run_id,
                entities=len(batch.entities),
                relations=len(batch.relations),
                validation_findings=len(validation_report.findings),
            )
            return ConstructionResult(
                batch=batch,
                validation_report=validation_report,
                property_graph=property_graph,
                rdf_graph=rdf_graph,
                evidence_graph=evidence_graph,
                metrics=metrics,
                run_manifest=manifest,
                quarantine_records=quarantine_records,
                human_review_items=human_review_items,
                skipped_sources=skipped_sources,
            )
        except Exception as exc:
            quarantine_records.extend(self._quarantine_exception(stage="workflow", error=exc))
            manifest = manifest.fail(exc, stage="workflow")
            self._write_manifest(manifest)
            self._log("exception", "workflow_failed", run_id=manifest.run_id)
            raise

    def _merge_component(
        self,
        batch: ConstructionBatch,
        *,
        component: Any,
        method_name: str,
        source: SourceRecord,
        stage: str,
        quarantine_records: list[QuarantineRecord],
    ) -> ConstructionBatch:
        component_name = getattr(component, "name", component.__class__.__name__)
        try:
            result = getattr(component, method_name)(source)
            self._log(
                "info",
                f"{stage}_completed",
                component=component_name,
                source_id=source.source_id,
            )
            return batch.merged(result)
        except Exception as exc:
            quarantine_records.extend(
                self._quarantine_exception(
                    stage=f"{stage}:{component_name}",
                    error=exc,
                    source=source,
                )
            )
            self._log(
                "exception",
                f"{stage}_failed",
                component=component_name,
                source_id=source.source_id,
            )
            if not self.continue_on_error:
                raise
            return batch

    def _redact_sources(self, sources: list[SourceRecord]) -> list[SourceRecord]:
        if self.pii_policy is None or not self.pii_policy.enabled:
            return sources
        return [self.pii_policy.redact_source_record(source) for source in sources]

    def _quarantine_exception(
        self,
        *,
        stage: str,
        error: BaseException,
        source: SourceRecord | None = None,
    ) -> list[QuarantineRecord]:
        if self.quarantine is None:
            return []
        return [
            self.quarantine.quarantine_exception(
                stage=stage,
                error=error,
                source=source,
            )
        ]

    def _write_manifest(self, manifest: RunManifest) -> None:
        if self.manifest_writer is not None:
            self.manifest_writer.write(manifest)

    def _log(self, level: str, event: str, **fields: Any) -> None:
        if self.structured_logger is None:
            return
        log_method = getattr(self.structured_logger, level)
        log_method(event, **fields)


__all__ = ["ConstructionResult", "ConstructionWorkflow"]
