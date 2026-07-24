"""One mandatory composed runtime for unstructured knowledge-graph publication."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.application.unstructured_pipeline import (
    PipelineResult,
    RuntimeProfile,
    UnstructuredPipeline,
    UnstructuredPipelineConfig,
)
from kg_construction.domain.models import ConstructionBatch, SourceRecord, canonical_json
from kg_construction.infrastructure.evaluation.claim_evaluations import (
    ClaimEvaluationSuite,
    HardCoverageReport,
    HardCoverageThresholds,
)
from kg_construction.infrastructure.extractors.model_backed.factory import (
    MandatoryProviderBundle,
    ModelBackendOverrides,
    build_mandatory_model_providers_from_lock,
)
from kg_construction.infrastructure.extractors.model_backed.identity import (
    LockedEmbeddingIdentityProvider,
)
from kg_construction.infrastructure.ontology.unstructured_governance import (
    FinalResourceShaclResult,
    GovernedResourceShaclGate,
    GovernedUnstructuredSchema,
)


class HeldOutClaimCase(BaseModel):
    """One immutable text/gold pair in the versioned benchmark contract."""

    model_config = ConfigDict(extra="allow", frozen=True)

    case_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    text: str = Field(min_length=1)

    def gold(self) -> dict[str, Any]:
        return self.model_dump(
            mode="json",
            exclude={"case_id", "source_name", "text"},
        )


class HeldOutClaimDataset(BaseModel):
    """Versioned, content-addressed held-out claim evaluation set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    thresholds: HardCoverageThresholds
    cases: tuple[HeldOutClaimCase, ...] = Field(min_length=1)
    sha256: str = Field(min_length=64, max_length=64)
    source_path: Path | None = None

    @classmethod
    def load(cls, path: Path | None = None) -> HeldOutClaimDataset:
        """Load either an explicitly reviewed path or the wheel-safe default."""

        if path is None:
            raw = files("kg_construction").joinpath(
                "resources/unstructured_claims_gold_v1.json"
            ).read_bytes()
            source_path = None
        else:
            if not path.is_file():
                raise FileNotFoundError(f"held-out claim gold does not exist: {path}")
            raw = path.read_bytes()
            source_path = path.resolve()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("held-out claim gold must contain one JSON object")
        return cls.model_validate(
            {
                **payload,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "source_path": source_path,
            }
        )


class HeldOutClaimCaseReport(BaseModel):
    """Hard-gate result for one held-out case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    prediction_sha256: str = Field(min_length=64, max_length=64)
    required_provider_ids: tuple[str, ...]
    participating_provider_ids: tuple[str, ...]
    provider_participation_passed: bool
    result: HardCoverageReport


class HeldOutClaimEvaluationReport(BaseModel):
    """Conjunctive result for every case in one versioned held-out dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    dataset_version: str
    dataset_sha256: str
    model_lock_sha256: str = Field(min_length=64, max_length=64)
    profile: str = Field(min_length=1)
    cases: tuple[HeldOutClaimCaseReport, ...]
    passed: bool


class MandatoryUnstructuredRuntime:
    """Compose locked extraction, identity, linking, ontology, SHACL, and gold eval."""

    name = "mandatory_composed_unstructured_runtime"
    technique = "governed_locked_unstructured_publication"
    schema_version = "v1"

    def __init__(
        self,
        *,
        pipeline: UnstructuredPipeline,
        providers: MandatoryProviderBundle,
        governed_schema: GovernedUnstructuredSchema,
        held_out_gold: HeldOutClaimDataset,
        model_lock_sha256: str,
    ) -> None:
        self.pipeline = pipeline
        self.providers = providers
        self.governed_schema = governed_schema
        self.held_out_gold = held_out_gold
        self.model_lock_sha256 = model_lock_sha256
        self.claim_evaluator = ClaimEvaluationSuite()
        self.claim_evaluation_report: HeldOutClaimEvaluationReport | None = None
        self.final_resource_shacl_gate = GovernedResourceShaclGate(governed_schema)
        self.final_resource_shacl_result: FinalResourceShaclResult | None = None

    @classmethod
    def from_lock(
        cls,
        lock_path: Path,
        *,
        cache_root: Path | None = None,
        governed_schema_path: Path | None = None,
        held_out_gold_path: Path | None = None,
        entity_labels: tuple[str, ...],
        ollama_base_url: str,
        ollama_retry_on_invalid_output: bool,
        ollama_max_retries: int,
        overrides: ModelBackendOverrides | None = None,
        profile: str | None = None,
        system: str | None = None,
        machine: str | None = None,
        cuda_available: bool | None = None,
    ) -> MandatoryUnstructuredRuntime:
        """Build every component from exact locked paths with no fallback."""

        schema = (
            GovernedUnstructuredSchema.from_yaml(governed_schema_path)
            if governed_schema_path is not None
            else GovernedUnstructuredSchema.from_package_resource()
        )
        undeclared_labels = set(entity_labels) - set(schema.entity_types)
        if undeclared_labels:
            raise ValueError(
                "model entity labels are absent from the governed schema: "
                f"{sorted(undeclared_labels)}"
            )
        providers = build_mandatory_model_providers_from_lock(
            lock_path,
            profile=profile,
            cache_root=cache_root,
            entity_labels=entity_labels,
            overrides=overrides,
            ollama_base_url=ollama_base_url,
            ollama_retry_on_invalid_output=ollama_retry_on_invalid_output,
            ollama_max_retries=ollama_max_retries,
            governed_schema_id=schema.schema_id,
            governed_entity_types=schema.entity_types,
            system=system,
            machine=machine,
            cuda_available=cuda_available,
        )
        pipeline = UnstructuredPipeline(
            config=UnstructuredPipelineConfig(profile=RuntimeProfile.MODEL_BACKED),
            mention_detector=providers.mention_detector,
            claim_extractor=providers.claim_extractor,
            identity_resolver=providers.identity_resolver,
            governed_schema=schema,
        )
        return cls(
            pipeline=pipeline,
            providers=providers,
            governed_schema=schema,
            held_out_gold=HeldOutClaimDataset.load(held_out_gold_path),
            model_lock_sha256=hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        )

    @property
    def mandatory_provider_ids(self) -> tuple[str, ...]:
        return self.providers.mandatory_runtime_participation_ids

    @property
    def pipeline_results(self) -> tuple[PipelineResult, ...]:
        return self.pipeline.pipeline_results

    @property
    def embedding_model(self) -> LockedEmbeddingIdentityProvider:
        """Return the exact locked model already used by identity resolution."""

        for provider in self.providers.identity_resolver.providers:
            if isinstance(provider, LockedEmbeddingIdentityProvider):
                return provider
        raise RuntimeError("mandatory identity ensemble has no locked embedding model")

    def reset_results(self) -> None:
        self.pipeline.reset_results()
        self.claim_evaluation_report = None
        self.final_resource_shacl_result = None

    def prepare_sources(self, sources: list[SourceRecord]) -> None:
        """Execute the versioned held-out suite in the same runtime invocation."""

        del sources
        reports: list[HeldOutClaimCaseReport] = []
        for case in self.held_out_gold.cases:
            source = SourceRecord.from_text(
                case.text,
                source_type="text",
                source_name=case.source_name,
            )
            result = self.pipeline.run(source)
            gate = self.claim_evaluator.hard_gate(
                result,
                case.gold(),
                self.held_out_gold.thresholds,
            )
            prediction_payload = result.model_dump(mode="json")
            required = (
                *self.providers.mention_detector.provider_ids,
                *self.providers.claim_extractor.provider_ids,
                *self.providers.identity_resolver.provider_ids,
            )
            participating = result.provider_participation_ids
            participation_passed = set(required) <= set(participating)
            reports.append(
                HeldOutClaimCaseReport(
                    case_id=case.case_id,
                    prediction_sha256=hashlib.sha256(
                        canonical_json(prediction_payload).encode("utf-8")
                    ).hexdigest(),
                    required_provider_ids=required,
                    participating_provider_ids=participating,
                    provider_participation_passed=participation_passed,
                    result=gate,
                )
            )
        self.claim_evaluation_report = HeldOutClaimEvaluationReport(
            dataset_id=self.held_out_gold.dataset_id,
            dataset_version=self.held_out_gold.version,
            dataset_sha256=self.held_out_gold.sha256,
            model_lock_sha256=self.model_lock_sha256,
            profile=self.providers.profile,
            cases=tuple(reports),
            passed=all(
                item.result.passed and item.provider_participation_passed
                for item in reports
            ),
        )
        if not self.claim_evaluation_report.passed:
            failures = {
                report.case_id: report.result.failures
                for report in reports
                if not report.result.passed
            }
            raise RuntimeError(
                f"versioned held-out claim evaluation failed: {failures}"
            )

    def extract(self, source: SourceRecord) -> ConstructionBatch:
        """Run all components and return only ontology/SHACL-qualified output."""

        if self.claim_evaluation_report is None:
            self.prepare_sources([source])
        batch = self.pipeline.extract(source)
        result = self.pipeline_results[-1]
        if (
            not result.shacl_executed
            or not result.shacl_conforms
            or result.governed_schema_sha256 != self.governed_schema.source_sha256
        ):
            raise RuntimeError(
                "unstructured source failed governed ontology/SHACL publication"
            )
        resolved = self.providers.resolution_ensemble.resolve(batch)
        resolution_participation = {
            str(row["provider_id"])
            for row in self.providers.resolution_ensemble.last_participation
            if row.get("executed") is True
        }
        participation = {
            *result.provider_participation_ids,
            *resolution_participation,
        }
        missing = set(self.mandatory_provider_ids) - participation
        if missing:
            raise RuntimeError(
                f"mandatory unstructured providers did not participate: {sorted(missing)}"
            )
        _final_rdf_graph, final_shacl = self.final_resource_shacl_gate.validate(
            resolved
        )
        self.final_resource_shacl_result = final_shacl
        if (
            not final_shacl.executed
            or not final_shacl.conforms
            or final_shacl.schema_sha256 != self.governed_schema.source_sha256
            or final_shacl.shacl_contract_sha256
            != self.final_resource_shacl_gate.shacl_contract_sha256
        ):
            raise RuntimeError(
                "post-promotion RDF resource graph failed governed SHACL: "
                f"{final_shacl.results_text}"
            )
        serialized_final_shacl = final_shacl.model_dump(mode="json")
        return resolved.model_copy(
            update={
                "metadata": {
                    **resolved.metadata,
                    "final_resource_shacl": serialized_final_shacl,
                    "mandatory_unstructured_runtime": {
                        "profile": self.providers.profile,
                        "provider_ids": list(self.mandatory_provider_ids),
                        "participating_provider_ids": sorted(participation),
                        "locked_artifact_paths": [
                            str(path) for path in self.providers.locked_artifact_paths
                        ],
                        "governed_schema_id": self.governed_schema.schema_id,
                        "governed_schema_version": self.governed_schema.version,
                        "governed_schema_sha256": self.governed_schema.source_sha256,
                        "held_out_dataset_id": self.held_out_gold.dataset_id,
                        "held_out_dataset_version": self.held_out_gold.version,
                        "held_out_dataset_sha256": self.held_out_gold.sha256,
                        "held_out_claim_evaluation_passed": bool(
                            self.claim_evaluation_report
                            and self.claim_evaluation_report.passed
                        ),
                        "source_shacl_executed": result.shacl_executed,
                        "source_shacl_conforms": result.shacl_conforms,
                        "final_resource_shacl_executed": final_shacl.executed,
                        "final_resource_shacl_conforms": final_shacl.conforms,
                        "final_rdf_graph_sha256": final_shacl.rdf_graph_sha256,
                        "final_shacl_contract_version": (
                            final_shacl.shacl_contract_version
                        ),
                        "final_shacl_contract_sha256": (
                            final_shacl.shacl_contract_sha256
                        ),
                        "final_resource_schema_sha256": final_shacl.schema_sha256,
                        "identity_nil_policy": result.identity_nil_policy,
                    },
                    "held_out_claim_evaluation": (
                        self.claim_evaluation_report.model_dump(mode="json")
                        if self.claim_evaluation_report is not None
                        else None
                    ),
                }
            }
        )


__all__ = [
    "HeldOutClaimCase",
    "HeldOutClaimCaseReport",
    "HeldOutClaimDataset",
    "HeldOutClaimEvaluationReport",
    "MandatoryUnstructuredRuntime",
]
