"""End-to-end product construction workflow orchestration."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kg_construction.certification.publication_contract import (
    TrustedPublicationBinding,
)
from kg_construction.domain.graph_objects import PropertyGraph, RDFGraph
from kg_construction.domain.modeling import (
    ControlledVocabulary,
    OntologyClass,
    OntologyPredicate,
    OntologySchema,
)
from kg_construction.domain.models import ConstructionBatch, SourceRecord, stable_id
from kg_construction.domain.validation import ValidationReport
from kg_construction.evaluation import (
    ConstructionReplayEvidence,
    GraphBoundIntendedUseEvaluator,
    IntendedUseEvaluationInputs,
    IntendedUseEvaluationReport,
    IntendedUseMetricsEvaluator,
    KnowledgeGraphQualityEvaluator,
    KnowledgeGraphQualityReport,
    QualityEvaluationInputs,
    RetryObservation,
    RuntimeReport,
    VersionedGraphEvaluationSuite,
    predicted_quality_mappings,
    property_graph_sha256,
)
from kg_construction.evaluation.metrics import ConstructionMetricsEvaluator
from kg_construction.evaluation.validation_metrics import (
    validation_failure_metrics,
)
from kg_construction.infrastructure.graph_builders import (
    EvidenceGraphBuilder,
    PropertyGraphBuilder,
    RDFGraphBuilder,
)
from kg_construction.infrastructure.ontology.shacl import ShaclValidator
from kg_construction.infrastructure.ontology.structured_ontology import (
    generate_shacl_shapes,
)
from kg_construction.infrastructure.ontology.validation import (
    CompositeValidator,
    EvidenceRequiredValidator,
    ReferentialIntegrityValidator,
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
from kg_construction.structured.resolved_plan import ResolvedModelingPlan
from kg_construction.unstructured.pipeline import PipelineResult
from kg_construction.unstructured.runtime import (
    HeldOutClaimEvaluationReport,
)


class PublicationGateReport(BaseModel):
    """Conjunctive decision required before any graph may be published."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    validation_passed: bool
    quality_passed: bool
    intended_use_passed: bool
    held_out_claim_passed: bool
    graph_bound_evaluation: bool
    certifiable_workload_backend: bool
    passed: bool
    failed_gates: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _conjunction_is_exact(self) -> PublicationGateReport:
        checks = {
            "validation": self.validation_passed,
            "quality": self.quality_passed,
            "intended_use": self.intended_use_passed,
            "held_out_claim": self.held_out_claim_passed,
            "graph_bound_evaluation": self.graph_bound_evaluation,
            "certifiable_workload_backend": self.certifiable_workload_backend,
        }
        failed = tuple(name for name, passed in checks.items() if not passed)
        if self.failed_gates != failed or self.passed != (not failed):
            raise ValueError("publication gate summary must equal its conjunction")
        return self


class ConstructionResult(BaseModel):
    """Complete result of one Data-to-KG workflow run."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    batch: ConstructionBatch
    validation_report: ValidationReport
    property_graph: PropertyGraph
    rdf_graph: RDFGraph
    evidence_graph: PropertyGraph
    metrics: dict[str, float] = Field(default_factory=dict)
    quality_report: KnowledgeGraphQualityReport
    intended_use_report: IntendedUseEvaluationReport
    run_manifest: RunManifest | None = None
    quarantine_records: list[QuarantineRecord] = Field(default_factory=list)
    human_review_items: list[HumanReviewItem] = Field(default_factory=list)
    skipped_sources: list[SourceRecord] = Field(default_factory=list)
    pipeline_results: tuple[PipelineResult, ...] = ()
    held_out_claim_evaluation: HeldOutClaimEvaluationReport | None = None
    publication_contract: TrustedPublicationBinding | None = None
    publication_gate: PublicationGateReport

    @property
    def release_eligible(self) -> bool:
        """True only for a passing graph with repository-owned trust evidence."""

        return (
            self.publication_gate.passed
            and self.publication_contract is not None
            and self.publication_contract.release_eligible
        )


class PublicationGateError(RuntimeError):
    """Raised when a strict workflow constructs a draft that cannot be published."""

    def __init__(self, result: ConstructionResult) -> None:
        self.result = result
        failed = ", ".join(result.publication_gate.failed_gates)
        quality_failures = [
            (
                f"{family_name}.{gate.metric}="
                f"{'not-executed' if gate.actual is None else f'{gate.actual:g}'} "
                f"(required {gate.operator} {gate.threshold:g})"
            )
            for family_name, family in result.quality_report.families.items()
            if family.applicable
            for gate in family.gates
            if not gate.passed
        ]
        detail = (
            f"; failed quality metrics: {', '.join(quality_failures)}"
            if "quality" in result.publication_gate.failed_gates
            and quality_failures
            else ""
        )
        super().__init__(
            f"knowledge graph publication gates failed: {failed}{detail}"
        )


class ConstructionWorkflow:
    """Run construction with every publication gate enforced.

    Non-publishable experiments must use the explicitly named
    :class:`DraftConstructionWorkflow` from this construction module.
    The production workflow intentionally exposes no flag that can disable its
    publication decision.
    """

    _publication_gates_required = True

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
        quality_evaluator: KnowledgeGraphQualityEvaluator | None = None,
        quality_inputs: QualityEvaluationInputs | None = None,
        intended_use_evaluator: IntendedUseMetricsEvaluator | None = None,
        intended_use_inputs: IntendedUseEvaluationInputs | None = None,
        graph_evaluation_suite: VersionedGraphEvaluationSuite | None = None,
        graph_bound_evaluator: GraphBoundIntendedUseEvaluator | None = None,
        publication_contract: TrustedPublicationBinding | None = None,
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
        self.quality_evaluator = quality_evaluator or KnowledgeGraphQualityEvaluator()
        self.quality_inputs = quality_inputs
        self.intended_use_evaluator = (
            intended_use_evaluator or IntendedUseMetricsEvaluator()
        )
        self.intended_use_inputs = intended_use_inputs
        self.graph_evaluation_suite = graph_evaluation_suite
        self.graph_bound_evaluator = (
            graph_bound_evaluator or GraphBoundIntendedUseEvaluator()
        )
        self.publication_contract = publication_contract
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

    def run(
        self,
        sources: list[SourceRecord],
        *,
        quality_inputs: QualityEvaluationInputs | None = None,
        intended_use_inputs: IntendedUseEvaluationInputs | None = None,
        graph_evaluation_suite: VersionedGraphEvaluationSuite | None = None,
    ) -> ConstructionResult:
        started_at = time.perf_counter()
        working_sources = self._redact_sources(sources)
        manifest = RunManifest.start(
            pipeline_name=self.pipeline_name,
            sources=working_sources,
            config_payload=self.config_payload,
            metadata={
                "publication_contract": (
                    self.publication_contract.model_dump(mode="json")
                    if self.publication_contract is not None
                    else None
                )
            },
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

            construction_phase_name = f"{self.pipeline_name}:construction"
            construction_phase_config = self._construction_phase_config()
            initial_construction_manifest = (
                RunManifest.start(
                    pipeline_name=construction_phase_name,
                    sources=processing_sources,
                    config_payload=construction_phase_config,
                )
                if self._publication_gates_required
                else None
            )

            for component in [*self.mappers, *self.extractors]:
                reset = getattr(component, "reset_results", None)
                if callable(reset):
                    reset()
                prepare = getattr(component, "prepare_sources", None)
                if callable(prepare):
                    prepare(processing_sources)

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
            pipeline_results = self._pipeline_results()
            held_out_claim_evaluation = self._held_out_claim_evaluation()
            replay_graph: PropertyGraph | None = None
            replay_evidence: ConstructionReplayEvidence | None = None
            operational_run_manifests: tuple[RunManifest, ...] = ()
            retry_observations: tuple[RetryObservation, ...] = ()
            if initial_construction_manifest is not None:
                initial_construction_manifest = initial_construction_manifest.complete(
                    batch=batch,
                    validation_report=validation_report,
                )
                replay_result = self._run_construction_replay(
                    processing_sources,
                    pipeline_name=construction_phase_name,
                    config_payload=construction_phase_config,
                )
                replay_manifest = replay_result.run_manifest
                if replay_manifest is None:
                    raise RuntimeError(
                        "construction replay did not produce a completed run manifest"
                    )
                replay_graph = replay_result.property_graph
                replay_evidence = ConstructionReplayEvidence(
                    initial_run=initial_construction_manifest,
                    replay_run=replay_manifest,
                    initial_graph_sha256=property_graph_sha256(property_graph),
                    replay_graph_sha256=property_graph_sha256(replay_graph),
                )
                operational_run_manifests = (
                    initial_construction_manifest,
                    replay_manifest,
                )
                retry_observations = (
                    RetryObservation(
                        operation_id=initial_construction_manifest.run_id,
                        attempt_count=1,
                        retry_count=0,
                    ),
                    RetryObservation(
                        operation_id=replay_manifest.run_id,
                        attempt_count=1,
                        retry_count=0,
                    ),
                )
            suite = graph_evaluation_suite or self.graph_evaluation_suite
            graph_execution = None
            supplied_quality_inputs: QualityEvaluationInputs | None
            if suite is not None:
                actual_profile = _construction_profile(batch, pipeline_results)
                if suite.construction_profile != actual_profile:
                    raise ValueError(
                        "evaluation suite construction_profile does not match "
                        f"the constructed graph: declared={suite.construction_profile}, "
                        f"actual={actual_profile}"
                    )
                intended_use_report, graph_execution = (
                    self.graph_bound_evaluator.evaluate(property_graph, suite)
                )
                predicted_coreference, predicted_links = (
                    predicted_quality_mappings(batch)
                )
                bound_quality_inputs = QualityEvaluationInputs(
                    evaluation_profile=suite.construction_profile,
                    expected_batch=(
                        suite.quality.expected_batch
                        if suite.quality.expected_batch.entities
                        or suite.quality.expected_batch.relations
                        else None
                    ),
                    predicted_coreference=(
                        predicted_coreference
                        if suite.quality.coreference_clusters
                        else None
                    ),
                    gold_coreference=(
                        suite.quality.coreference_clusters or None
                    ),
                    predicted_links=(
                        predicted_links if suite.quality.entity_links else None
                    ),
                    gold_links=suite.quality.entity_links or None,
                    predicted_answers=graph_execution.predicted_answers,
                    gold_answers=graph_execution.gold_answers,
                    vector_rankings=graph_execution.vector_rankings,
                    vector_relevant=graph_execution.vector_relevant,
                )
                if quality_inputs is not None or self.quality_inputs is not None:
                    raise ValueError(
                        "graph-bound evaluation cannot be combined with raw "
                        "QualityEvaluationInputs"
                    )
                supplied_quality_inputs = bound_quality_inputs
            else:
                intended_use_report = self.intended_use_evaluator.evaluate(
                    intended_use_inputs or self.intended_use_inputs
                )
                supplied_quality_inputs = quality_inputs or self.quality_inputs
            resolved_quality_inputs = self._resolve_quality_inputs(
                batch=batch,
                property_graph=property_graph,
                rdf_graph=rdf_graph,
                validation_report=validation_report,
                started_at=started_at,
                supplied=supplied_quality_inputs,
                pipeline_results=pipeline_results,
                observed_replay_graph=replay_graph,
                replay_evidence=replay_evidence,
                operational_run_manifests=operational_run_manifests,
                retry_observations=retry_observations,
            )
            quality_report = self.quality_evaluator.evaluate(
                batch,
                inputs=resolved_quality_inputs,
            )
            held_out_claim_passed = (
                not pipeline_results
                or (
                    held_out_claim_evaluation is not None
                    and held_out_claim_evaluation.passed
                )
            )
            gate_checks = {
                "validation": validation_report.valid,
                "quality": quality_report.passed,
                "intended_use": intended_use_report.passed,
                "held_out_claim": held_out_claim_passed,
                "graph_bound_evaluation": intended_use_report.graph_bound,
                "certifiable_workload_backend": bool(
                    graph_execution is not None
                    and graph_execution.certifiable_backend
                ),
            }
            failed_gates = tuple(
                name for name, passed in gate_checks.items() if not passed
            )
            publication_gate = PublicationGateReport(
                validation_passed=gate_checks["validation"],
                quality_passed=gate_checks["quality"],
                intended_use_passed=gate_checks["intended_use"],
                held_out_claim_passed=gate_checks["held_out_claim"],
                graph_bound_evaluation=gate_checks["graph_bound_evaluation"],
                certifiable_workload_backend=gate_checks[
                    "certifiable_workload_backend"
                ],
                passed=not failed_gates,
                failed_gates=failed_gates,
            )
            metrics = self.evaluator.evaluate(batch)
            metrics.update(validation_failure_metrics(batch, validation_report))
            metrics.update(quality_report.metric_payload())
            metrics["quality_passed"] = float(quality_report.passed)
            for family_name, family in quality_report.families.items():
                metrics[f"quality_{family_name}_executed"] = float(family.executed)
                metrics[f"quality_{family_name}_passed"] = float(family.passed)
            metrics["intended_use_passed"] = float(intended_use_report.passed)
            metrics["publication_passed"] = float(publication_gate.passed)
            for use_name, use_report in intended_use_report.families.items():
                metrics[f"intended_use_{use_name}_executed"] = float(
                    use_report.executed
                )
                metrics[f"intended_use_{use_name}_passed"] = float(use_report.passed)
                metrics.update(
                    {
                        f"intended_use_{use_name}_{name}": value
                        for name, value in use_report.metrics.items()
                    }
                )
            metrics["quarantine_record_count"] = float(len(quarantine_records))
            metrics["human_review_queue_count"] = float(len(human_review_items))
            metrics["skipped_source_count"] = float(len(skipped_sources))

            manifest = manifest.complete(
                batch=batch,
                validation_report=validation_report,
                metadata={
                    "quarantine_records": len(quarantine_records),
                    "human_review_items": len(human_review_items),
                    "skipped_sources": len(skipped_sources),
                    "quality_passed": quality_report.passed,
                    "quality_missing_families": quality_report.missing_families,
                    "intended_use_passed": intended_use_report.passed,
                    "missing_intended_uses": intended_use_report.missing_uses,
                    "held_out_claim_passed": held_out_claim_passed,
                    "publication_passed": publication_gate.passed,
                    "publication_failed_gates": publication_gate.failed_gates,
                    "publication_contract": (
                        self.publication_contract.model_dump(mode="json")
                        if self.publication_contract is not None
                        else None
                    ),
                },
            )
            result = ConstructionResult(
                batch=batch,
                validation_report=validation_report,
                property_graph=property_graph,
                rdf_graph=rdf_graph,
                evidence_graph=evidence_graph,
                metrics=metrics,
                quality_report=quality_report,
                intended_use_report=intended_use_report,
                run_manifest=manifest,
                quarantine_records=quarantine_records,
                human_review_items=human_review_items,
                skipped_sources=skipped_sources,
                pipeline_results=pipeline_results,
                held_out_claim_evaluation=held_out_claim_evaluation,
                publication_contract=self.publication_contract,
                publication_gate=publication_gate,
            )
            if self._publication_gates_required and not publication_gate.passed:
                raise PublicationGateError(result)
            if self.watermark_store is not None:
                self.incremental_sync_policy.commit(
                    processing_sources,
                    self.watermark_store,
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
            return result
        except Exception as exc:
            quarantine_records.extend(self._quarantine_exception(stage="workflow", error=exc))
            manifest = manifest.fail(exc, stage="workflow")
            self._write_manifest(manifest)
            self._log("exception", "workflow_failed", run_id=manifest.run_id)
            raise

    def _resolve_quality_inputs(
        self,
        *,
        batch: ConstructionBatch,
        property_graph: PropertyGraph,
        rdf_graph: RDFGraph,
        validation_report: ValidationReport,
        started_at: float,
        supplied: QualityEvaluationInputs | None,
        pipeline_results: tuple[PipelineResult, ...],
        observed_replay_graph: PropertyGraph | None,
        replay_evidence: ConstructionReplayEvidence | None,
        operational_run_manifests: tuple[RunManifest, ...],
        retry_observations: tuple[RetryObservation, ...],
    ) -> QualityEvaluationInputs:
        """Fill every construction-measurable input and preserve caller gold judgments."""

        inputs = (supplied or QualityEvaluationInputs()).model_copy(deep=True)
        plans = _resolved_plans(batch)
        ontology = inputs.ontology or _merged_ontology(plans)
        policy = inputs.policy or (plans[0] if len(plans) == 1 else None)
        elapsed = max(time.perf_counter() - started_at, 0.0)
        processed_items = max(
            1,
            len(batch.source_records)
            + len(batch.entities)
            + len(batch.relations)
            + len(batch.facts),
        )
        runtime = inputs.runtime_report or RuntimeReport(
            processed_items=processed_items,
            latency_seconds=elapsed,
            throughput=(processed_items / elapsed if elapsed else 0.0),
            retry_observations=retry_observations,
            metadata=(
                {"retry_observation_scope": "top_level_construction_invocation"}
                if retry_observations
                else {}
            ),
        )
        if retry_observations and not runtime.retry_observations:
            runtime = runtime.model_copy(
                update={
                    "retry_observations": retry_observations,
                    "metadata": {
                        **runtime.metadata,
                        "retry_observation_scope": (
                            "top_level_construction_invocation"
                        ),
                    },
                }
            )

        shacl_executed = False
        shacl_violation_count: int | None = None
        shacl_checked_count: int | None = None
        governed_schema_id: str | None = None
        governed_schema_version: str | None = None
        governed_schema_sha256: str | None = None
        if ontology is not None:
            shacl_result = ShaclValidator().validate_rdf_graph(
                rdf_graph,
                shacl_turtle=generate_shacl_shapes(
                    ontology,
                    namespace=ontology.namespace,
                ),
            )
            shacl_executed = True
            shacl_violation_count = _shacl_violation_count(
                shacl_result.results_graph_turtle
            )
            shacl_checked_count = max(
                1,
                len(batch.entities) + len(batch.relations) + len(batch.facts),
            )
        elif pipeline_results:
            schema_tuples = {
                (
                    result.governed_schema_id,
                    result.governed_schema_version,
                    result.governed_schema_sha256,
                )
                for result in pipeline_results
            }
            if len(schema_tuples) != 1:
                raise ValueError(
                    "unstructured pipeline results must use one governed schema"
                )
            governed_schema_id, governed_schema_version, governed_schema_sha256 = (
                schema_tuples.pop()
            )
            shacl_executed = all(
                result.shacl_executed for result in pipeline_results
            )
            promoted_claim_ids = {
                claim.claim_id
                for result in pipeline_results
                for claim in result.promoted
            }
            promoted_shacl_results = tuple(
                {
                    item.claim_id: item
                    for result in pipeline_results
                    for item in result.claim_shacl_results
                    if item.claim_id in promoted_claim_ids
                }.values()
            )
            shacl_violation_count = sum(
                int(not item.conforms)
                for item in promoted_shacl_results
            )
            shacl_checked_count = len(promoted_shacl_results)

        return inputs.model_copy(
            update={
                "property_graph": property_graph,
                "rdf_graph": rdf_graph,
                "validation_report": validation_report,
                "ontology": ontology,
                "policy": policy,
                "source_records": inputs.source_records or _source_rows(batch),
                "shacl_executed": shacl_executed,
                "shacl_violation_count": shacl_violation_count,
                "shacl_checked_count": shacl_checked_count,
                "governed_schema_id": governed_schema_id,
                "governed_schema_version": governed_schema_version,
                "governed_schema_sha256": governed_schema_sha256,
                "replay_graph": inputs.replay_graph or observed_replay_graph,
                "replay_evidence": inputs.replay_evidence or replay_evidence,
                "runtime_report": runtime,
                "operational_run_manifests": (
                    inputs.operational_run_manifests
                    or operational_run_manifests
                ),
            }
        )

    def _construction_phase_config(self) -> dict[str, object]:
        """Return the stable configuration identity used by both replay runs."""

        def component_identity(component: object) -> dict[str, object]:
            component_type = type(component)
            config = getattr(component, "config", None)
            if isinstance(config, BaseModel):
                config_payload: object | None = config.model_dump(mode="json")
            elif isinstance(config, dict | list | tuple | str | int | float | bool):
                config_payload = config
            else:
                config_payload = None
            return {
                "type": (
                    f"{component_type.__module__}."
                    f"{component_type.__qualname__}"
                ),
                "name": str(
                    getattr(component, "name", component_type.__name__)
                ),
                "config": config_payload,
            }

        return {
            "declared_config": self.config_payload,
            "mappers": [
                component_identity(component) for component in self.mappers
            ],
            "extractors": [
                component_identity(component) for component in self.extractors
            ],
            "resolvers": [
                component_identity(component) for component in self.resolvers
            ],
            "validators": [
                component_identity(component)
                for component in self.validator.validators
            ],
            "property_graph_builder": component_identity(
                self.property_graph_builder
            ),
            "rdf_graph_builder": component_identity(self.rdf_graph_builder),
            "evidence_graph_builder": component_identity(
                self.evidence_graph_builder
            ),
            "continue_on_error": self.continue_on_error,
        }

    def _run_construction_replay(
        self,
        sources: list[SourceRecord],
        *,
        pipeline_name: str,
        config_payload: dict[str, object],
    ) -> ConstructionResult:
        """Execute a second full construction pass without recursive replay."""

        replay = DraftConstructionWorkflow(
            mappers=list(self.mappers),
            extractors=list(self.extractors),
            resolvers=list(self.resolvers),
            validators=list(self.validator.validators),
            property_graph_builder=self.property_graph_builder,
            rdf_graph_builder=self.rdf_graph_builder,
            evidence_graph_builder=self.evidence_graph_builder,
            pipeline_name=pipeline_name,
            config_payload=config_payload,
            continue_on_error=self.continue_on_error,
        )
        return replay.run(sources)

    def _pipeline_results(self) -> tuple[PipelineResult, ...]:
        results: list[PipelineResult] = []
        for extractor in self.extractors:
            raw = getattr(extractor, "pipeline_results", ())
            if isinstance(raw, tuple | list):
                results.extend(
                    item for item in raw if isinstance(item, PipelineResult)
                )
        return tuple(results)

    def _held_out_claim_evaluation(
        self,
    ) -> HeldOutClaimEvaluationReport | None:
        for extractor in self.extractors:
            report = getattr(extractor, "claim_evaluation_report", None)
            if isinstance(report, HeldOutClaimEvaluationReport):
                return report
            delegate = getattr(extractor, "_delegate", None)
            report = getattr(delegate, "claim_evaluation_report", None)
            if isinstance(report, HeldOutClaimEvaluationReport):
                return report
        return None

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


class DraftConstructionWorkflow(ConstructionWorkflow):
    """Internal, explicitly non-publishable workflow for research and tests."""

    _publication_gates_required = False


def _source_rows(batch: ConstructionBatch) -> list[dict[str, object]] | None:
    rows = [dict(row) for source in batch.source_records for row in source.records]
    return rows or None


def _construction_profile(
    batch: ConstructionBatch,
    pipeline_results: tuple[PipelineResult, ...],
) -> str:
    unstructured_versions = {
        result.document.version for result in pipeline_results
    }
    has_unstructured = bool(unstructured_versions)
    has_other = any(
        source.source_id not in unstructured_versions
        for source in batch.source_records
    )
    if has_unstructured and has_other:
        return "mixed"
    if has_unstructured:
        return "unstructured"
    return "structured"


def _resolved_plans(batch: ConstructionBatch) -> list[ResolvedModelingPlan]:
    payload = batch.metadata.get("resolved_modeling_plans", {})
    if not isinstance(payload, dict):
        return []
    plans: list[ResolvedModelingPlan] = []
    for plan_id, raw in sorted(payload.items()):
        if not isinstance(raw, dict):
            raise ValueError(
                f"resolved modelling plan {plan_id!r} must be a serialized object"
            )
        plans.append(ResolvedModelingPlan.model_validate(raw))
    return plans


def _merged_ontology(plans: list[ResolvedModelingPlan]) -> OntologySchema | None:
    if not plans:
        return None
    ontologies = [plan.to_ontology_schema() for plan in plans]
    classes: dict[str, OntologyClass] = {}
    predicates: dict[tuple[str, str, str], OntologyPredicate] = {}
    vocabularies: dict[str, ControlledVocabulary] = {}
    for ontology in ontologies:
        for ontology_class in ontology.classes:
            previous_class = classes.get(ontology_class.name)
            if (
                previous_class is not None
                and previous_class.role != ontology_class.role
            ):
                raise ValueError(
                    "conflicting class definitions for "
                    f"{ontology_class.name!r} across resolved plans"
                )
            if previous_class is None or (
                not previous_class.key_properties
                and ontology_class.key_properties
            ):
                classes[ontology_class.name] = ontology_class
        for predicate in ontology.predicates:
            predicate_key = (
                predicate.name,
                predicate.domain,
                predicate.kind,
            )
            previous_predicate = predicates.get(predicate_key)
            if previous_predicate is not None and previous_predicate.model_dump(
                exclude={"description"}
            ) != predicate.model_dump(exclude={"description"}):
                raise ValueError(
                    "conflicting predicate definitions for "
                    f"{predicate.domain!r}.{predicate.name!r} across resolved plans"
                )
            predicates.setdefault(predicate_key, predicate)
        for vocabulary in ontology.controlled_vocabularies:
            previous_vocabulary = vocabularies.get(vocabulary.name)
            if previous_vocabulary is not None and previous_vocabulary != vocabulary:
                raise ValueError(
                    f"conflicting controlled vocabularies for {vocabulary.name!r}"
                )
            vocabularies.setdefault(vocabulary.name, vocabulary)
    return OntologySchema(
        schema_id=stable_id("merged-ontology", *(plan.plan_id for plan in plans)),
        version=plans[0].version,
        namespace=plans[0].namespace,
        classes=[classes[name] for name in sorted(classes)],
        predicates=[predicates[key] for key in sorted(predicates)],
        controlled_vocabularies=[
            vocabularies[name] for name in sorted(vocabularies)
        ],
    )


def _shacl_violation_count(results_graph_turtle: str) -> int:
    from rdflib import Graph
    from rdflib.namespace import RDF, SH

    graph = Graph()
    graph.parse(data=results_graph_turtle, format="turtle")
    return sum(1 for _ in graph.subjects(RDF.type, SH.ValidationResult))


__all__ = [
    "ConstructionResult",
    "ConstructionWorkflow",
    "DraftConstructionWorkflow",
    "PublicationGateError",
    "PublicationGateReport",
]
