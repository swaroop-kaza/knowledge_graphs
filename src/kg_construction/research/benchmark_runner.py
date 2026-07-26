"""Comparative technique evaluation and selection for research workflows."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from kg_construction.certification.registry import (
    PluginRegistry,
    TechniqueDescriptor,
    default_registry,
)
from kg_construction.construction.profiling import DatasetProfile, DatasetProfiler
from kg_construction.construction.workflow import (
    ConstructionResult,
    DraftConstructionWorkflow,
)
from kg_construction.domain.graph_objects import PropertyGraph, RDFGraph
from kg_construction.domain.models import ConstructionBatch, SourceRecord
from kg_construction.evaluation.accuracy_metrics import AccuracyMetrics
from kg_construction.evaluation.coverage_metrics import CoverageMetrics
from kg_construction.evaluation.entity_resolution_metrics import (
    EntityResolutionMetrics,
)
from kg_construction.evaluation.gold_standards import (
    GoldBenchmarkEvaluator,
    GoldBenchmarkFixture,
    GoldEntityResolutionFixture,
    GoldRetrievalFixture,
)
from kg_construction.evaluation.operational_metrics import (
    OperationalMetrics,
    OperationalMetricsRecorder,
    RuntimeReport,
)
from kg_construction.evaluation.retrieval_metrics import RetrievalMetrics
from kg_construction.evaluation.schema_quality import SchemaQualityEvaluator
from kg_construction.evaluation.validation_metrics import ValidationMetrics
from kg_construction.infrastructure.extractors.gazetteer import GazetteerEntityExtractor
from kg_construction.infrastructure.extractors.regex import RegexEntityExtractor
from kg_construction.infrastructure.graph_builders.evidence_graph import EvidenceGraphBuilder
from kg_construction.infrastructure.graph_builders.property_graph import PropertyGraphBuilder
from kg_construction.infrastructure.graph_builders.rdf_graph import RDFGraphBuilder
from kg_construction.infrastructure.loaders import (
    APIResponseLoader,
    ChatTranscriptLoader,
    EmailLoader,
    ExcelLoader,
    MarkdownLoader,
    PDFLoader,
    ReportLoader,
    XMLLoader,
)
from kg_construction.infrastructure.loaders.csv_loader import CsvLoader
from kg_construction.infrastructure.loaders.json_loader import JsonLoader
from kg_construction.infrastructure.loaders.text_loader import TextLoader
from kg_construction.infrastructure.mapping.direct_table_mapper import DirectTableMapper
from kg_construction.infrastructure.mapping.semi_structured import JsonPathMapper
from kg_construction.infrastructure.ontology.validation import (
    EvidenceRequiredValidator,
    ValidationReport,
)
from kg_construction.infrastructure.resolution.exact import ExactKeyEntityResolver
from kg_construction.research.recommendation_engine import (
    RecommendationEngine,
    RecommendationSummary,
)
from kg_construction.research.scoring import (
    PenaltyWeights,
    ResearchScreeningThresholds,
    ScoringEngine,
    ScoringWeights,
    TechniqueScore,
)
from kg_construction.research.technique_selector import ApplicabilityResult, TechniqueSelector
from kg_construction.structured.source_catalog import CatalogTableMapper

DEFAULT_GAZETTEER = {
    "Organization": ["Platform Team", "Research", "Compiler", "Flight", "ML"],
    "Incident": ["INCIDENT-1042"],
    "System": ["Recommendation API", "Feature Store"],
}

TIE_BREAKING_POLICY = [
    "final_score_desc",
    "research_screening_thresholds_pass_desc",
    "latency_asc",
    "maintainability_desc",
    "technique_name_asc",
]

SENSITIVE_SETTING_TOKENS = (
    "apikey",
    "credential",
    "credentials",
    "password",
    "secret",
    "token",
)


class EvaluationSettings(BaseModel):
    """Settings controlling which techniques are benchmarked."""

    model_config = ConfigDict(extra="forbid")

    run_all_applicable_techniques: bool = Field(
        default=True,
        description=(
            "Reserved compatibility setting; technique selection currently follows the "
            "CLI selection plus force_all_techniques."
        ),
    )
    force_all_techniques: bool = False
    gold_dataset_path: str | None = None
    human_review_sample_size: int = Field(
        default=100,
        ge=0,
        description=(
            "Reserved for a future human-review workflow; it does not currently sample "
            "or alter benchmark scores."
        ),
    )
    bootstrap_iterations: int = Field(
        default=1000,
        ge=0,
        description=(
            "Reserved for future confidence intervals; current reports label score "
            "uncertainty as not_estimated."
        ),
    )
    allow_external_services: bool = False
    deterministic_runtime: bool = True
    fixed_latency_seconds: float = 0.0
    fixed_throughput: float = 1000.0
    fixed_memory_peak_mb: float = 0.0
    fixed_graph_write_time_seconds: float = 0.0
    fixed_ollama_prompt_tokens: int = 0
    fixed_ollama_output_tokens: int = 0
    fixed_ollama_cost_proxy: float = 0.0


class BenchmarkHardGateSettings(BaseModel):
    """Gold/task gates used to classify one non-authoritative comparison."""

    model_config = ConfigDict(extra="forbid")

    require_gold_dataset: bool = True
    maximum_dangling_edge_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    maximum_duplicate_edge_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    minimum_evidence_coverage: float = Field(default=0.95, ge=0.0, le=1.0)
    maximum_validation_failure_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    maximum_false_merge_rate: float = Field(default=0.02, ge=0.0, le=1.0)
    minimum_entity_f1: float = Field(default=0.80, ge=0.0, le=1.0)
    minimum_span_f1: float = Field(default=0.80, ge=0.0, le=1.0)
    minimum_relation_f1: float = Field(default=0.80, ge=0.0, le=1.0)
    minimum_triple_exact_match: float = Field(default=0.80, ge=0.0, le=1.0)
    minimum_pairwise_er_recall: float = Field(default=0.80, ge=0.0, le=1.0)
    minimum_er_record_id_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    minimum_retrieval_recall_at_k: float = Field(default=0.70, ge=0.0, le=1.0)
    minimum_retrieval_prediction_completeness: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )
    # Schema/instance quality gates. Opt-in: default off so existing smoke and gold
    # benchmarks are unaffected. Enable for policy-driven semantic (structured) graphs.
    enforce_schema_quality_gates: bool = False
    maximum_orphan_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    maximum_duplicate_node_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    maximum_categorical_duplication_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    maximum_schema_consistency_violation_count: float = Field(default=0.0, ge=0.0)


class TechniqueEvaluationConfig(BaseModel):
    """YAML config for benchmark and recommendation runs."""

    model_config = ConfigDict(extra="forbid")

    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    applicability_threshold: float = 0.40
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    penalties: PenaltyWeights = Field(default_factory=PenaltyWeights)
    screening_thresholds: ResearchScreeningThresholds = Field(
        default_factory=ResearchScreeningThresholds,
        validation_alias=AliasChoices(
            "screening_thresholds",
            "production_thresholds",
        ),
    )
    hard_gates: BenchmarkHardGateSettings = Field(default_factory=BenchmarkHardGateSettings)
    llm: dict[str, Any] = Field(
        default_factory=lambda: {"provider": "ollama", "model": "qwen3:1.7b-q4_K_M"}
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> TechniqueEvaluationConfig:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(payload)


class GoldDatasetManifest(BaseModel):
    """Versioned description of the gold fixtures used by one benchmark dataset."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    benchmark_id: str
    version: str
    description: str = ""
    domain: str = "general"
    split: str = "test"
    publication_date: str | None = None
    license: str | None = None
    input_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_inventory: list[str] = Field(default_factory=list)
    extraction_fixture: str | None = None
    entity_resolution_fixture: str | None = None
    retrieval_fixture: str | None = None
    citations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GoldEvaluationSummary(BaseModel):
    """Gold fixtures applied to a candidate and tasks that were actually evaluated."""

    model_config = ConfigDict(extra="forbid")

    configured: bool = False
    benchmark_id: str | None = None
    benchmark_version: str | None = None
    dataset_hash: str | None = None
    evaluated_tasks: list[str] = Field(default_factory=list)
    unavailable_tasks: list[str] = Field(default_factory=list)
    support: dict[str, int] = Field(default_factory=dict)


class HardGateResult(BaseModel):
    """Outcome of one explicit research benchmark gate."""

    model_config = ConfigDict(extra="forbid")

    gate: str
    metric: str
    operator: Literal["<=", ">=", "required"]
    threshold: float | bool
    actual: float | bool | None = None
    passed: bool = False
    evaluated: bool = True
    reason: str = ""


class CandidateGraphResult(BaseModel):
    """Candidate graph and evaluation payload for one technique."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    technique_name: str
    pipeline_name: str
    batch: ConstructionBatch | None = None
    property_graph: PropertyGraph | None = None
    rdf_graph: RDFGraph | None = None
    evidence_graph: PropertyGraph | None = None
    validation_report: ValidationReport | None = None
    evaluation_report: dict[str, float] = Field(default_factory=dict)
    runtime_report: RuntimeReport = Field(default_factory=RuntimeReport)
    applicability_score: float = 0.0
    gold_evaluation: GoldEvaluationSummary = Field(default_factory=GoldEvaluationSummary)
    hard_gates: list[HardGateResult] = Field(default_factory=list)
    research_gate_passed: bool = False
    error: str | None = None

    @property
    def publication_eligible(self) -> bool:
        """Deprecated alias; a research gate never authorizes publication."""

        return self.research_gate_passed


class BenchmarkRunMetadata(BaseModel):
    """Reproducibility metadata for one benchmark execution."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    timestamp: str
    config_hash: str
    git_commit: str
    git_tree_state: Literal["clean", "dirty", "unknown"] = "unknown"
    source_tree_hash: str = "unknown"
    input_hash: str
    expected_input_hash: str | None = None
    input_contract_verified: bool | None = None
    source_inventory: list[str] = Field(default_factory=list)
    gold_dataset_path: str | None = None
    gold_dataset_hash: str | None = None
    benchmark_id: str | None = None
    benchmark_version: str | None = None
    source_type: str | None = None
    techniques: str
    deterministic_runtime: bool
    tie_breaking_policy: list[str] = Field(default_factory=list)
    model_settings: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, str] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _LoadedGoldDataset:
    """Validated in-memory gold bundle loaded from a fixture or manifest."""

    path: Path
    dataset_hash: str
    benchmark_id: str
    benchmark_version: str
    expected_input_hash: str | None = None
    source_inventory: tuple[Path, ...] = ()
    extraction: GoldBenchmarkFixture | None = None
    entity_resolution: GoldEntityResolutionFixture | None = None
    retrieval: GoldRetrievalFixture | None = None


class TechniqueEvaluationReport(BaseModel):
    """Complete benchmark report."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    generated_at: str
    run_metadata: BenchmarkRunMetadata
    input_path: str
    dataset_profile: DatasetProfile
    applicability_results: list[ApplicabilityResult] = Field(default_factory=list)
    candidates: list[CandidateGraphResult] = Field(default_factory=list)
    ranked_scores: list[TechniqueScore] = Field(default_factory=list)
    recommendations: RecommendationSummary

    @property
    def best_ranked_technique(self) -> str | None:
        """Return the top non-authoritative weighted ranking."""

        return self.recommendations.best_single_technique.technique_name

    @property
    def best_technique(self) -> str | None:
        """Backward-compatible alias for :attr:`best_ranked_technique`."""

        return self.best_ranked_technique


class TechniqueEvaluationRunner:
    """Profile data, run candidate pipelines, score them, and recommend a winner."""

    name = "technique_evaluation_runner"

    def __init__(
        self,
        *,
        config: TechniqueEvaluationConfig | None = None,
        registry: PluginRegistry | None = None,
    ) -> None:
        self.config = config or TechniqueEvaluationConfig()
        self.registry = registry or default_registry()
        self.profiler = DatasetProfiler()
        self.selector = TechniqueSelector(
            registry=self.registry,
            threshold=self.config.applicability_threshold,
        )
        self.scoring = ScoringEngine(weights=self.config.weights, penalties=self.config.penalties)
        self.recommender = RecommendationEngine(self.config.screening_thresholds)

    def run(
        self,
        *,
        input_path: str | Path,
        source_type: str | None = None,
        techniques: str = "applicable",
        output_dir: str | Path | None = None,
    ) -> TechniqueEvaluationReport:
        generated_at = datetime.now(UTC).isoformat()
        gold_dataset = self._load_gold_dataset(self.config.evaluation.gold_dataset_path)
        evaluated_input_hash: str | None = None
        if gold_dataset is not None and gold_dataset.source_inventory:
            source_paths, evaluated_input_hash = self._validated_source_inventory(
                Path(input_path),
                gold_dataset,
            )
            sources = [
                source
                for source_path in source_paths
                for source in self._load_sources(source_path, None)
            ]
        else:
            sources = self._load_sources(input_path, source_type)
            if gold_dataset is not None and gold_dataset.expected_input_hash is not None:
                evaluated_input_hash = self._input_hash(Path(input_path))
                self._require_expected_input_hash(
                    expected=gold_dataset.expected_input_hash,
                    actual=evaluated_input_hash,
                )
        profile = self.profiler.profile_sources(sources, input_path=str(input_path))
        force_all = techniques == "all" or self.config.evaluation.force_all_techniques
        applicability = self.selector.select(profile, force_all=force_all)
        selected_names = self._selected_names(applicability, techniques)
        candidate_specs = self._candidate_specs(sources, selected_names, profile)
        candidates = [
            self._run_candidate(spec, sources, profile, gold_dataset)
            for spec in candidate_specs
        ]
        ranked_scores = self._score_candidates(candidates)
        recommendations = self.recommender.recommend(
            ranked_scores,
            research_gate_passing_techniques={
                candidate.technique_name
                for candidate in candidates
                if candidate.research_gate_passed
            },
            evaluated_gold_tasks={
                task
                for candidate in candidates
                for task in candidate.gold_evaluation.evaluated_tasks
            },
        )
        report = TechniqueEvaluationReport(
            generated_at=generated_at,
            run_metadata=self._build_run_metadata(
                input_path=Path(input_path),
                source_type=source_type,
                techniques=techniques,
                timestamp=generated_at,
                gold_dataset=gold_dataset,
                evaluated_input_hash=evaluated_input_hash,
            ),
            input_path=self._portable_report_path(Path(input_path)),
            dataset_profile=profile,
            applicability_results=applicability,
            candidates=candidates,
            ranked_scores=ranked_scores,
            recommendations=recommendations,
        )
        if output_dir is not None:
            self._save_candidate_outputs(report, Path(output_dir))
        return report

    def _load_sources(self, input_path: str | Path, source_type: str | None) -> list[SourceRecord]:
        path = Path(input_path)
        if path.is_dir():
            sources: list[SourceRecord] = []
            for child in sorted(path.iterdir()):
                if child.is_file() and child.suffix.lower() in {
                    ".chat",
                    ".csv",
                    ".eml",
                    ".json",
                    ".md",
                    ".markdown",
                    ".pdf",
                    ".report",
                    ".txt",
                    ".xlsx",
                    ".xlsm",
                    ".xml",
                }:
                    sources.extend(self._load_sources(child, None))
            return sources
        inferred = source_type or path.suffix.lower().lstrip(".")
        if inferred == "csv":
            return CsvLoader().load(path)
        if inferred == "json":
            return JsonLoader().load(path)
        if inferred in {"xlsx", "xlsm", "excel"}:
            return ExcelLoader().load(path)
        if inferred == "xml":
            return XMLLoader().load(path)
        if inferred in {"api", "api_response"}:
            return APIResponseLoader().load(path)
        if inferred in {"md", "markdown"}:
            return MarkdownLoader().load(path)
        if inferred == "pdf":
            return PDFLoader().load(path)
        if inferred == "report":
            return ReportLoader().load(path)
        if inferred in {"eml", "email"}:
            return EmailLoader().load(path)
        if inferred in {"chat", "chat_transcript", "transcript"}:
            return ChatTranscriptLoader().load(path)
        return TextLoader().load(path)

    @staticmethod
    def _selected_names(applicability: list[ApplicabilityResult], techniques: str) -> set[str]:
        if techniques in {"applicable", "all"}:
            return {item.technique_name for item in applicability}
        return {name.strip() for name in techniques.split(",") if name.strip()}

    def _candidate_specs(
        self,
        sources: list[SourceRecord],
        selected_names: set[str],
        profile: DatasetProfile,
    ) -> list[dict[str, Any]]:
        has_records = any(source.records for source in sources)
        has_text = any(source.content for source in sources) or profile.text_density > 0
        specs: list[dict[str, Any]] = []
        if has_records and self._should_run("semantic_table_mapper", selected_names):
            specs.append(
                {
                    "technique_name": "semantic_table_mapper",
                    "pipeline_name": "semantic_mapping",
                    "mappers": [CatalogTableMapper()],
                    "extractors": [],
                }
            )
        if has_records and self._should_run("direct_table_mapper", selected_names):
            specs.append(
                {
                    "technique_name": "direct_table_mapper",
                    "pipeline_name": "direct_mapping",
                    "mappers": [DirectTableMapper()],
                    "extractors": [],
                }
            )
        if has_records and self._should_run("jsonpath_mapper", selected_names):
            specs.append(
                {
                    "technique_name": "jsonpath_mapper",
                    "pipeline_name": "jsonpath_mapping",
                    "mappers": [JsonPathMapper()],
                    "extractors": [],
                }
            )
        if has_text and self._should_run("regex_entity_extractor", selected_names):
            specs.append(
                {
                    "technique_name": "regex_entity_extractor",
                    "pipeline_name": "regex_rules",
                    "mappers": [],
                    "extractors": [RegexEntityExtractor()],
                }
            )
        if has_text and self._should_run("gazetteer_entity_extractor", selected_names):
            specs.append(
                {
                    "technique_name": "gazetteer_entity_extractor",
                    "pipeline_name": "gazetteer_rules",
                    "mappers": [],
                    "extractors": [GazetteerEntityExtractor(DEFAULT_GAZETTEER)],
                }
            )
        if has_text and (
            self._should_run("hybrid_rules_gazetteer_regex", selected_names)
            or self._should_run("hybrid_rules_ml_ollama", selected_names)
            or len(specs) < 3
        ):
            specs.append(
                {
                    "technique_name": "hybrid_rules_gazetteer_regex",
                    "pipeline_name": "hybrid_regex_plus_gazetteer",
                    "mappers": [],
                    "extractors": [RegexEntityExtractor(), GazetteerEntityExtractor(DEFAULT_GAZETTEER)],
                }
            )
        if has_records and has_text and self._should_run("hybrid_structured_then_text", selected_names):
            specs.append(
                {
                    "technique_name": "hybrid_structured_then_text",
                    "pipeline_name": "hybrid_direct_mapping_plus_regex",
                    "mappers": [DirectTableMapper()],
                    "extractors": [RegexEntityExtractor()],
                }
            )
        return specs

    @staticmethod
    def _should_run(technique_name: str, selected_names: set[str]) -> bool:
        return technique_name in selected_names or not selected_names

    def _run_candidate(
        self,
        spec: dict[str, Any],
        sources: list[SourceRecord],
        profile: DatasetProfile,
        gold_dataset: _LoadedGoldDataset | None,
    ) -> CandidateGraphResult:
        processed_items = max(1, profile.record_count or profile.source_count)
        result: ConstructionResult | None = None
        error: str | None = None
        with OperationalMetricsRecorder(processed_items=processed_items) as recorder:
            try:
                workflow = DraftConstructionWorkflow(
                    mappers=spec["mappers"],
                    extractors=spec["extractors"],
                    resolvers=[ExactKeyEntityResolver()],
                    validators=[EvidenceRequiredValidator()],
                    property_graph_builder=PropertyGraphBuilder(),
                    rdf_graph_builder=RDFGraphBuilder(),
                    evidence_graph_builder=EvidenceGraphBuilder(),
                )
                result = workflow.run(sources)
            except Exception as exc:
                error = str(exc)
        runtime_report = self._runtime_report(recorder.report)
        if error or result is None:
            return CandidateGraphResult(
                technique_name=str(spec["technique_name"]),
                pipeline_name=str(spec["pipeline_name"]),
                runtime_report=runtime_report,
                error=error or "Technique did not produce a construction result.",
            )
        metrics, gold_evaluation = self._evaluate_result(
            result,
            runtime_report,
            profile,
            gold_dataset,
        )
        if self.config.hard_gates.enforce_schema_quality_gates:
            metrics = {
                **metrics,
                **SchemaQualityEvaluator().structural_metrics(result.property_graph),
            }
        hard_gates = self._evaluate_hard_gates(metrics, gold_evaluation)
        return CandidateGraphResult(
            technique_name=str(spec["technique_name"]),
            pipeline_name=str(spec["pipeline_name"]),
            batch=result.batch,
            property_graph=result.property_graph,
            rdf_graph=result.rdf_graph,
            evidence_graph=result.evidence_graph,
            validation_report=result.validation_report,
            evaluation_report=metrics,
            runtime_report=runtime_report,
            applicability_score=self._applicability_score(str(spec["technique_name"]), profile),
            gold_evaluation=gold_evaluation,
            hard_gates=hard_gates,
            # A no-gold ranking cannot pass even these non-authoritative
            # comparison gates.
            research_gate_passed=(
                bool(gold_evaluation.evaluated_tasks)
                and all(gate.passed for gate in hard_gates)
            ),
        )

    def _runtime_report(self, report: RuntimeReport) -> RuntimeReport:
        if not self.config.evaluation.deterministic_runtime:
            return report
        settings = self.config.evaluation
        metadata = dict(report.metadata)
        metadata["runtime_mode"] = "deterministic"
        return report.model_copy(
            update={
                "latency_seconds": settings.fixed_latency_seconds,
                "throughput": settings.fixed_throughput,
                "memory_peak_mb": settings.fixed_memory_peak_mb,
                "graph_write_time_seconds": settings.fixed_graph_write_time_seconds,
                "ollama_prompt_tokens": settings.fixed_ollama_prompt_tokens,
                "ollama_output_tokens": settings.fixed_ollama_output_tokens,
                "ollama_cost_proxy": settings.fixed_ollama_cost_proxy,
                "metadata": metadata,
            }
        )

    def _evaluate_result(
        self,
        result: ConstructionResult,
        runtime_report: RuntimeReport,
        profile: DatasetProfile,
        gold_dataset: _LoadedGoldDataset | None,
    ) -> tuple[dict[str, float], GoldEvaluationSummary]:
        metrics: dict[str, float] = {}
        metrics.update(AccuracyMetrics().evaluate(result.batch))
        metrics.update(
            ValidationMetrics().evaluate(
                result.batch,
                result.validation_report,
                property_graph=result.property_graph,
                rdf_graph=result.rdf_graph,
            )
        )
        metrics.update(
            CoverageMetrics().evaluate(
                result.batch,
                estimated_entities=max(profile.expected_graph_size, 1),
                estimated_relations=max(int(profile.relation_density_score * profile.expected_graph_size), 1),
            )
        )
        if gold_dataset is None or gold_dataset.entity_resolution is None:
            metrics.update(EntityResolutionMetrics().evaluate(result.batch))
        metrics.update(RetrievalMetrics().evaluate())
        gold_evaluation = self._apply_gold_metrics(result, metrics, gold_dataset)
        metrics.update(OperationalMetrics().evaluate(runtime_report))
        metrics["estimated_precision"] = (
            metrics["evidence_coverage"]
            * metrics["schema_validity_score"]
            * (1.0 - metrics["unsupported_extraction_rate"])
        )
        metrics["entity_count"] = float(len(result.batch.entities))
        metrics["relation_count"] = float(len(result.batch.relations))
        return metrics, gold_evaluation

    def _apply_gold_metrics(
        self,
        result: ConstructionResult,
        metrics: dict[str, float],
        gold_dataset: _LoadedGoldDataset | None,
    ) -> GoldEvaluationSummary:
        if gold_dataset is None:
            return GoldEvaluationSummary()

        evaluated_tasks: list[str] = []
        unavailable_tasks: list[str] = []
        support: dict[str, int] = {}
        evaluator = GoldBenchmarkEvaluator()

        if (gold := gold_dataset.extraction) is not None:
            declared_tasks = gold.model_fields_set
            if "entity_spans" in declared_tasks:
                entity_score = evaluator.evaluate_entities(result.batch, gold)
                span_score = evaluator.evaluate_entity_spans(result.batch, gold)
                metrics.update(
                    {
                        "entity_precision": entity_score.precision,
                        "entity_recall": entity_score.recall,
                        "entity_f1": entity_score.f1,
                        "span_f1": span_score.f1,
                        "gold_entity_true_positive": float(entity_score.true_positive),
                        "gold_entity_false_positive": float(entity_score.false_positive),
                        "gold_entity_false_negative": float(entity_score.false_negative),
                    }
                )
                evaluated_tasks.extend(["entities", "entity_spans"])
                support["entities"] = len(gold.entity_spans)
                support["entity_spans"] = len(gold.entity_spans)
            if "relations" in declared_tasks:
                relation_score = evaluator.evaluate_relation_mentions(result.batch, gold)
                metrics.update(
                    {
                        "relation_precision": relation_score.precision,
                        "relation_recall": relation_score.recall,
                        "relation_f1": relation_score.f1,
                        "gold_relation_true_positive": float(relation_score.true_positive),
                        "gold_relation_false_positive": float(relation_score.false_positive),
                        "gold_relation_false_negative": float(relation_score.false_negative),
                    }
                )
                evaluated_tasks.append("relations")
                support["relations"] = len(gold.relations)
            if "triples" in declared_tasks:
                triple_score = evaluator.evaluate_triples(result.batch, gold)
                metrics.update(
                    {
                        "triple_exact_match": triple_score.f1,
                        "relaxed_triple_match": triple_score.f1,
                        "gold_triple_precision": triple_score.precision,
                        "gold_triple_recall": triple_score.recall,
                        "gold_triple_true_positive": float(triple_score.true_positive),
                        "gold_triple_false_positive": float(triple_score.false_positive),
                        "gold_triple_false_negative": float(triple_score.false_negative),
                    }
                )
                evaluated_tasks.append("triples")
                support["triples"] = len(gold.triples)
            if not declared_tasks.intersection(
                {"entity_spans", "relations", "triples"}
            ):
                unavailable_tasks.append("no_extraction_tasks_declared")

        if (gold_er := gold_dataset.entity_resolution) is not None:
            gold_pairs = gold_er.gold_pairs()
            gold_record_ids = gold_er.record_ids()
            metrics.update(
                EntityResolutionMetrics().evaluate(
                    result.batch,
                    gold_pairs=gold_pairs,
                    gold_record_ids=gold_record_ids,
                    record_id_property=gold_er.evaluation_record_id_property,
                )
            )
            evaluated_tasks.append("entity_resolution")
            support["entity_resolution_pairs"] = len(gold_pairs)
            support["entity_resolution_gold_records"] = len(gold_record_ids)
            support["entity_resolution_predicted_records"] = int(
                metrics["er_predicted_record_support"]
            )
            support["entity_resolution_record_id_overlap"] = int(
                metrics["er_record_id_overlap_support"]
            )
            metrics["gold_entity_resolution_closed_world"] = float(
                gold_er.closed_world
            )

        if (gold_retrieval := gold_dataset.retrieval) is not None:
            if not gold_retrieval.cases:
                unavailable_tasks.append("empty_retrieval_fixture")
            else:
                retrieval_metrics = self._evaluate_gold_retrieval(result, gold_retrieval)
                metrics.update(retrieval_metrics)
                evaluated_tasks.append("retrieval")
                support["retrieval_queries"] = len(gold_retrieval.cases)
                support["retrieval_predictions"] = int(
                    metrics["retrieval_predicted_query_count"]
                )
                support["retrieval_missing_predictions"] = int(
                    metrics["retrieval_missing_query_count"]
                )

        metrics["gold_evaluated_task_count"] = float(len(evaluated_tasks))
        metrics["gold_entities_evaluated"] = float("entities" in evaluated_tasks)
        metrics["gold_relations_evaluated"] = float("relations" in evaluated_tasks)
        metrics["gold_triples_evaluated"] = float("triples" in evaluated_tasks)
        metrics["gold_entity_resolution_evaluated"] = float(
            "entity_resolution" in evaluated_tasks
        )
        metrics["gold_retrieval_evaluated"] = float("retrieval" in evaluated_tasks)

        return GoldEvaluationSummary(
            configured=True,
            benchmark_id=gold_dataset.benchmark_id,
            benchmark_version=gold_dataset.benchmark_version,
            dataset_hash=gold_dataset.dataset_hash,
            evaluated_tasks=evaluated_tasks,
            unavailable_tasks=unavailable_tasks,
            support=support,
        )

    @staticmethod
    def _evaluate_gold_retrieval(
        result: ConstructionResult,
        gold: GoldRetrievalFixture,
    ) -> dict[str, float]:
        raw_predictions = result.batch.metadata.get("retrieval_predictions")
        if not isinstance(raw_predictions, dict):
            raw_predictions = {}

        per_case: list[dict[str, float]] = []
        predicted_query_count = 0
        for case in gold.cases:
            raw_value = raw_predictions.get(case.query_id)
            if isinstance(raw_value, dict):
                raw_prediction: dict[str, Any] = raw_value
                prediction_present = True
                predicted_query_count += 1
            else:
                raw_prediction = {}
                prediction_present = False
            retrieved = [str(item) for item in raw_prediction.get("retrieved_context_ids", [])]
            predicted_path = [str(item) for item in raw_prediction.get("path", [])]
            predicted_citations = {
                str(item) for item in raw_prediction.get("citation_ids", [])
            }
            answer = str(raw_prediction.get("answer", ""))
            path_matches = (
                prediction_present
                and bool(case.expected_path)
                and predicted_path == case.expected_path
            )
            is_multi_hop = len(case.expected_path) > 3
            citation_matches = (
                prediction_present
                and bool(case.expected_citation_ids)
                and set(case.expected_citation_ids).issubset(predicted_citations)
            )
            answer_matches = (
                prediction_present
                and bool(case.expected_answer_contains)
                and all(
                    expected.casefold() in answer.casefold()
                    for expected in case.expected_answer_contains
                )
            )
            per_case.append(
                RetrievalMetrics().evaluate(
                    retrieved=retrieved,
                    relevant=set(case.relevant_context_ids),
                    useful_paths=int(path_matches),
                    retrieved_paths=int(bool(predicted_path)),
                    correct_multi_hop_answers=int(is_multi_hop and answer_matches),
                    total_multi_hop_questions=int(is_multi_hop),
                    correct_citations=int(citation_matches),
                    total_answers=1,
                    faithful_answers=int(answer_matches and citation_matches),
                )
            )
        if not per_case:
            return {
                **RetrievalMetrics().evaluate(),
                "retrieval_gold_query_count": 0.0,
                "retrieval_predicted_query_count": 0.0,
                "retrieval_missing_query_count": 0.0,
                "retrieval_prediction_completeness": 0.0,
            }
        metric_names = per_case[0]
        metrics = {
            name: sum(row[name] for row in per_case) / len(per_case)
            for name in metric_names
        }
        total_queries = len(gold.cases)
        metrics.update(
            {
                "retrieval_gold_query_count": float(total_queries),
                "retrieval_predicted_query_count": float(predicted_query_count),
                "retrieval_missing_query_count": float(
                    total_queries - predicted_query_count
                ),
                "retrieval_prediction_completeness": (
                    predicted_query_count / total_queries
                    if total_queries
                    else 0.0
                ),
            }
        )
        return metrics

    def _evaluate_hard_gates(
        self,
        metrics: dict[str, float],
        gold: GoldEvaluationSummary,
    ) -> list[HardGateResult]:
        settings = self.config.hard_gates
        gates = [
            self._maximum_gate(
                "referential_integrity",
                "dangling_edge_rate",
                metrics.get("dangling_edge_rate", 0.0),
                settings.maximum_dangling_edge_rate,
            ),
            self._maximum_gate(
                "semantic_edge_uniqueness",
                "duplicate_edge_rate",
                metrics.get("duplicate_edge_rate", 0.0),
                settings.maximum_duplicate_edge_rate,
            ),
            self._minimum_gate(
                "evidence_traceability",
                "evidence_coverage",
                metrics.get("evidence_coverage", 0.0),
                settings.minimum_evidence_coverage,
            ),
            self._maximum_gate(
                "validation_failures",
                "validation_failure_rate",
                metrics.get("validation_failure_rate", 1.0),
                settings.maximum_validation_failure_rate,
            ),
        ]
        if "entity_resolution" in gold.evaluated_tasks:
            gates.extend(
                [
                    self._maximum_gate(
                        "false_merges",
                        "false_merge_rate",
                        metrics.get("false_merge_rate", 1.0),
                        settings.maximum_false_merge_rate,
                    ),
                    self._minimum_gate(
                        "entity_resolution_recall",
                        "pairwise_er_recall",
                        metrics.get("pairwise_er_recall", 0.0),
                        settings.minimum_pairwise_er_recall,
                    ),
                ]
            )
            if metrics.get("gold_entity_resolution_closed_world", 0.0) > 0.0:
                gates.append(
                    self._minimum_gate(
                        "entity_resolution_record_id_coverage",
                        "er_record_id_symmetric_coverage",
                        metrics.get("er_record_id_symmetric_coverage", 0.0),
                        settings.minimum_er_record_id_coverage,
                    )
                )
        if "entities" in gold.evaluated_tasks:
            gates.append(
                self._minimum_gate(
                    "entity_extraction_accuracy",
                    "entity_f1",
                    metrics.get("entity_f1", 0.0),
                    settings.minimum_entity_f1,
                )
            )
        if "entity_spans" in gold.evaluated_tasks:
            gates.append(
                self._minimum_gate(
                    "entity_span_accuracy",
                    "span_f1",
                    metrics.get("span_f1", 0.0),
                    settings.minimum_span_f1,
                )
            )
        if "relations" in gold.evaluated_tasks:
            gates.append(
                self._minimum_gate(
                    "relation_extraction_accuracy",
                    "relation_f1",
                    metrics.get("relation_f1", 0.0),
                    settings.minimum_relation_f1,
                )
            )
        if "triples" in gold.evaluated_tasks:
            gates.append(
                self._minimum_gate(
                    "triple_accuracy",
                    "triple_exact_match",
                    metrics.get("triple_exact_match", 0.0),
                    settings.minimum_triple_exact_match,
                )
            )
        if "retrieval" in gold.evaluated_tasks:
            gates.extend(
                [
                    self._minimum_gate(
                        "retrieval_prediction_completeness",
                        "retrieval_prediction_completeness",
                        metrics.get("retrieval_prediction_completeness", 0.0),
                        settings.minimum_retrieval_prediction_completeness,
                    ),
                    self._minimum_gate(
                        "retrieval_recall",
                        "retrieval_recall_at_k",
                        metrics.get("retrieval_recall_at_k", 0.0),
                        settings.minimum_retrieval_recall_at_k,
                    ),
                ]
            )
        if settings.enforce_schema_quality_gates:
            gates.append(
                self._maximum_gate(
                    "schema_connectivity", "orphan_rate",
                    metrics.get("orphan_rate", 0.0), settings.maximum_orphan_rate,
                )
            )
            gates.append(
                self._maximum_gate(
                    "schema_node_conciseness", "duplicate_node_rate",
                    metrics.get("duplicate_node_rate", 0.0), settings.maximum_duplicate_node_rate,
                )
            )
            gates.append(
                self._maximum_gate(
                    "schema_consistency", "consistency_violation_count",
                    metrics.get("consistency_violation_count", 0.0),
                    settings.maximum_schema_consistency_violation_count,
                )
            )
            if "categorical_duplication_rate" in metrics:
                gates.append(
                    self._maximum_gate(
                        "categorical_conciseness", "categorical_duplication_rate",
                        metrics["categorical_duplication_rate"],
                        settings.maximum_categorical_duplication_rate,
                    )
                )
        gates.append(
            HardGateResult(
                gate="gold_correctness",
                metric="gold_dataset_configured",
                operator="required",
                threshold=settings.require_gold_dataset,
                actual=bool(gold.evaluated_tasks),
                passed=bool(gold.evaluated_tasks) or not settings.require_gold_dataset,
                reason=(
                    "At least one compatible gold task was evaluated."
                    if gold.evaluated_tasks
                    else "No compatible gold task was evaluated; this is a smoke run only."
                ),
            )
        )
        return gates

    @staticmethod
    def _maximum_gate(
        gate: str,
        metric: str,
        actual: float,
        threshold: float,
    ) -> HardGateResult:
        return HardGateResult(
            gate=gate,
            metric=metric,
            operator="<=",
            threshold=threshold,
            actual=actual,
            passed=actual <= threshold,
        )

    @staticmethod
    def _minimum_gate(
        gate: str,
        metric: str,
        actual: float,
        threshold: float,
    ) -> HardGateResult:
        return HardGateResult(
            gate=gate,
            metric=metric,
            operator=">=",
            threshold=threshold,
            actual=actual,
            passed=actual >= threshold,
        )

    def _score_candidates(self, candidates: list[CandidateGraphResult]) -> list[TechniqueScore]:
        scores: list[TechniqueScore] = []
        for candidate in candidates:
            if candidate.error:
                continue
            descriptor = self._descriptor(candidate.technique_name)
            score = self.scoring.score(
                technique_name=candidate.technique_name,
                pipeline_name=candidate.pipeline_name,
                metrics=candidate.evaluation_report,
                descriptor=descriptor,
                applicability_score=candidate.applicability_score,
            )
            scores.append(score)
        return sorted(scores, key=self._ranking_key)

    def _ranking_key(self, score: TechniqueScore) -> tuple[float, bool, float, float, str]:
        return (
            -round(score.final_score, 12),
            not self._passes_screening_thresholds(score),
            round(score.metrics.get("latency", 0.0), 12),
            -round(score.dimensions.get("maintainability", 0.0), 12),
            score.technique_name,
        )

    def _passes_screening_thresholds(self, score: TechniqueScore) -> bool:
        metrics = score.metrics
        unsupported_extraction_rate = metrics.get(
            "unsupported_extraction_rate",
            metrics.get("hallucination_risk_proxy", metrics.get("hallucination_rate", 1.0)),
        )
        thresholds = self.config.screening_thresholds
        return (
            metrics.get("evidence_coverage", 0.0) >= thresholds.evidence_coverage
            and metrics.get("schema_validity_score", 0.0) >= thresholds.schema_validity
            and unsupported_extraction_rate <= thresholds.unsupported_extraction_threshold
            and metrics.get("validation_failure_rate", 1.0) <= thresholds.validation_failure_rate
        )

    def _descriptor(self, technique_name: str) -> TechniqueDescriptor | None:
        try:
            return self.registry.get(technique_name)
        except KeyError:
            return None

    def _applicability_score(self, technique_name: str, profile: DatasetProfile) -> float:
        descriptor = self._descriptor(technique_name)
        if descriptor is None:
            return 0.55
        return self.selector.score(descriptor, profile).score

    @staticmethod
    def _save_candidate_outputs(report: TechniqueEvaluationReport, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for candidate in report.candidates:
            if candidate.property_graph is None:
                continue
            safe_name = candidate.technique_name.replace("/", "_")
            (output_dir / f"{safe_name}.property_graph.json").write_text(
                candidate.property_graph.to_json(),
                encoding="utf-8",
            )
            if candidate.rdf_graph is not None:
                (output_dir / f"{safe_name}.rdf.json").write_text(
                    json.dumps(candidate.rdf_graph.to_dict(), indent=2, sort_keys=True),
                    encoding="utf-8",
                )

    @classmethod
    def _load_gold_dataset(cls, configured_path: str | None) -> _LoadedGoldDataset | None:
        if configured_path is None:
            return None
        path = Path(configured_path)
        if not path.exists():
            raise ValueError(f"Configured gold dataset does not exist: {path}")
        if path.is_dir():
            manifests = [
                candidate
                for candidate in (
                    path / "benchmark_manifest.yaml",
                    path / "benchmark_manifest.yml",
                    path / "benchmark_manifest.json",
                    path / "manifest.yaml",
                    path / "manifest.yml",
                    path / "manifest.json",
                )
                if candidate.exists()
            ]
            if len(manifests) != 1:
                raise ValueError(
                    "A gold dataset directory must contain exactly one benchmark manifest "
                    "named benchmark_manifest.{yaml,yml,json} or manifest.{yaml,yml,json}."
                )
            path = manifests[0]

        payload = cls._read_gold_payload(path)
        if "benchmark_id" in payload and "version" in payload:
            return cls._load_gold_manifest(path, payload)
        return cls._load_direct_gold_fixture(path, payload)

    @staticmethod
    def _read_gold_payload(path: Path) -> dict[str, Any]:
        if path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Gold dataset must be a JSON/YAML object: {path}")
        return payload

    @classmethod
    def _load_gold_manifest(
        cls,
        path: Path,
        payload: dict[str, Any],
    ) -> _LoadedGoldDataset:
        manifest = GoldDatasetManifest.model_validate(payload)
        fixture_paths = {
            "extraction": cls._resolve_fixture_path(path, manifest.extraction_fixture),
            "entity_resolution": cls._resolve_fixture_path(
                path,
                manifest.entity_resolution_fixture,
            ),
            "retrieval": cls._resolve_fixture_path(path, manifest.retrieval_fixture),
        }
        if not any(fixture_paths.values()):
            raise ValueError(f"Gold manifest has no fixture paths: {path}")
        extraction_path = fixture_paths["extraction"]
        er_path = fixture_paths["entity_resolution"]
        retrieval_path = fixture_paths["retrieval"]
        source_inventory = tuple(
            cls._resolve_inventory_path(path, item)
            for item in manifest.source_inventory
        )
        if len(set(source_inventory)) != len(source_inventory):
            raise ValueError("Gold manifest source_inventory contains duplicate paths.")
        return _LoadedGoldDataset(
            path=path,
            dataset_hash=cls._combined_file_hash(
                [path, *(item for item in fixture_paths.values() if item is not None)]
            ),
            benchmark_id=manifest.benchmark_id,
            benchmark_version=manifest.version,
            expected_input_hash=manifest.input_hash,
            source_inventory=source_inventory,
            extraction=(
                GoldBenchmarkFixture.from_json_file(extraction_path)
                if extraction_path is not None
                else None
            ),
            entity_resolution=(
                GoldEntityResolutionFixture.from_json_file(er_path)
                if er_path is not None
                else None
            ),
            retrieval=(
                GoldRetrievalFixture.from_json_file(retrieval_path)
                if retrieval_path is not None
                else None
            ),
        )

    @staticmethod
    def _resolve_inventory_path(manifest_path: Path, value: str) -> Path:
        inventory_path = Path(value)
        if inventory_path.is_absolute():
            raise ValueError(
                "Gold manifest source_inventory entries must be relative paths: "
                f"{value!r}."
            )
        resolved = (manifest_path.parent / inventory_path).resolve()
        try:
            resolved.relative_to(manifest_path.parent.resolve())
        except ValueError as exc:
            raise ValueError(
                "Gold manifest source_inventory entries cannot escape the manifest "
                f"directory: {value!r}."
            ) from exc
        return resolved

    @staticmethod
    def _resolve_fixture_path(manifest_path: Path, value: str | None) -> Path | None:
        if value is None:
            return None
        fixture_path = Path(value)
        if not fixture_path.is_absolute():
            fixture_path = manifest_path.parent / fixture_path
        if not fixture_path.is_file():
            raise ValueError(f"Gold fixture referenced by manifest does not exist: {fixture_path}")
        return fixture_path

    @classmethod
    def _load_direct_gold_fixture(
        cls,
        path: Path,
        payload: dict[str, Any],
    ) -> _LoadedGoldDataset:
        fixture_id = str(payload.get("fixture_id", path.stem))
        fixture_version = str(dict(payload.get("metadata", {})).get("version", "fixture"))
        dataset_hash = cls._combined_file_hash([path])
        if "contexts" in payload and "cases" in payload:
            return _LoadedGoldDataset(
                path=path,
                dataset_hash=dataset_hash,
                benchmark_id=fixture_id,
                benchmark_version=fixture_version,
                retrieval=GoldRetrievalFixture.model_validate(payload),
            )
        if "records" in payload:
            return _LoadedGoldDataset(
                path=path,
                dataset_hash=dataset_hash,
                benchmark_id=fixture_id,
                benchmark_version=fixture_version,
                entity_resolution=GoldEntityResolutionFixture.model_validate(payload),
            )
        if any(key in payload for key in ("entity_spans", "relations", "triples")):
            return _LoadedGoldDataset(
                path=path,
                dataset_hash=dataset_hash,
                benchmark_id=fixture_id,
                benchmark_version=fixture_version,
                extraction=GoldBenchmarkFixture.model_validate(payload),
            )
        raise ValueError(
            "Gold fixture schema is not recognized. Expected extraction, entity-resolution, "
            f"retrieval, or a benchmark manifest: {path}"
        )

    @staticmethod
    def _combined_file_hash(paths: list[Path]) -> str:
        digest = hashlib.sha256()
        for path in sorted(paths, key=lambda item: item.name):
            digest.update(path.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    def _build_run_metadata(
        self,
        *,
        input_path: Path,
        source_type: str | None,
        techniques: str,
        timestamp: str,
        gold_dataset: _LoadedGoldDataset | None,
        evaluated_input_hash: str | None,
    ) -> BenchmarkRunMetadata:
        config_hash = self._config_hash()
        input_hash = evaluated_input_hash or self._input_hash(input_path)
        run_seed = json.dumps(
            {
                "timestamp": timestamp,
                "config_hash": config_hash,
                "input_hash": input_hash,
                "gold_dataset_hash": gold_dataset.dataset_hash if gold_dataset else None,
                "techniques": techniques,
            },
            sort_keys=True,
        )
        run_id = hashlib.sha256(run_seed.encode("utf-8")).hexdigest()[:16]
        git_commit, git_tree_state = self._git_metadata()
        return BenchmarkRunMetadata(
            run_id=run_id,
            timestamp=timestamp,
            config_hash=config_hash,
            git_commit=git_commit,
            git_tree_state=git_tree_state,
            source_tree_hash=self._source_tree_hash(),
            input_hash=input_hash,
            expected_input_hash=(
                gold_dataset.expected_input_hash
                if gold_dataset
                else None
            ),
            input_contract_verified=(
                input_hash == gold_dataset.expected_input_hash
                if gold_dataset and gold_dataset.expected_input_hash is not None
                else None
            ),
            source_inventory=(
                [
                    path.relative_to(gold_dataset.path.parent.resolve()).as_posix()
                    for path in gold_dataset.source_inventory
                ]
                if gold_dataset
                else []
            ),
            gold_dataset_path=(
                self._portable_report_path(gold_dataset.path)
                if gold_dataset
                else None
            ),
            gold_dataset_hash=gold_dataset.dataset_hash if gold_dataset else None,
            benchmark_id=gold_dataset.benchmark_id if gold_dataset else None,
            benchmark_version=gold_dataset.benchmark_version if gold_dataset else None,
            source_type=source_type,
            techniques=techniques,
            deterministic_runtime=self.config.evaluation.deterministic_runtime,
            tie_breaking_policy=TIE_BREAKING_POLICY,
            model_settings={
                "llm": self._sanitize_model_settings(self.config.llm),
                "allow_external_services": self.config.evaluation.allow_external_services,
            },
            environment=self._environment_metadata(),
        )

    @classmethod
    def _validated_source_inventory(
        cls,
        input_path: Path,
        gold_dataset: _LoadedGoldDataset,
    ) -> tuple[tuple[Path, ...], str]:
        source_paths = gold_dataset.source_inventory
        missing = [path for path in source_paths if not path.is_file()]
        if missing:
            raise ValueError(
                "Gold manifest source inventory contains missing files: "
                f"{[path.name for path in missing]!r}."
            )

        resolved_input = input_path.resolve()
        if resolved_input.is_dir():
            outside = []
            for path in source_paths:
                try:
                    path.relative_to(resolved_input)
                except ValueError:
                    outside.append(path)
            if outside:
                raise ValueError(
                    "Gold manifest source inventory is not contained in the configured "
                    f"input directory: {[path.name for path in outside]!r}."
                )
        elif len(source_paths) != 1 or source_paths[0] != resolved_input:
            raise ValueError(
                "A file input must exactly match the single source_inventory entry in "
                "the gold manifest."
            )

        actual_hash = cls._inventory_input_hash(
            source_paths,
            root=gold_dataset.path.parent.resolve(),
        )
        if gold_dataset.expected_input_hash is not None:
            cls._require_expected_input_hash(
                expected=gold_dataset.expected_input_hash,
                actual=actual_hash,
            )
        return source_paths, actual_hash

    @staticmethod
    def _inventory_input_hash(paths: tuple[Path, ...], *, root: Path) -> str:
        digest = hashlib.sha256()
        for path in sorted(paths):
            relative_path = path.relative_to(root).as_posix()
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    @staticmethod
    def _require_expected_input_hash(*, expected: str, actual: str) -> None:
        if actual != expected:
            raise ValueError(
                "Benchmark input hash does not match the gold manifest contract: "
                f"expected {expected}, got {actual}."
            )

    @classmethod
    def _sanitize_model_settings(cls, value: Any, *, key: str = "") -> Any:
        """Remove credentials from reproducibility metadata before serialization."""

        normalized_key = key.casefold().replace("-", "_")
        key_parts = set(normalized_key.split("_"))
        if "api_key" in normalized_key or key_parts.intersection(
            SENSITIVE_SETTING_TOKENS
        ):
            return "[REDACTED]"
        if isinstance(value, dict):
            return {
                str(item_key): cls._sanitize_model_settings(
                    item_value,
                    key=str(item_key),
                )
                for item_key, item_value in value.items()
            }
        if isinstance(value, list):
            return [cls._sanitize_model_settings(item) for item in value]
        if isinstance(value, tuple):
            return [cls._sanitize_model_settings(item) for item in value]
        return value

    @staticmethod
    def _portable_report_path(path: Path) -> str:
        """Avoid publishing absolute workstation paths in benchmark reports."""

        if not path.is_absolute():
            return path.as_posix()
        resolved = path.resolve()
        try:
            return resolved.relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return f"<external>/{resolved.name}"

    @staticmethod
    def _environment_metadata() -> dict[str, str]:
        try:
            package_version = version("kg-construction")
        except PackageNotFoundError:
            package_version = "source-tree"
        return {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "package_version": package_version,
            "executable": Path(sys.executable).name,
        }

    def _config_hash(self) -> str:
        config_payload = self.config.model_dump(mode="json")
        config_payload["llm"] = self._sanitize_model_settings(
            config_payload.get("llm", {})
        )
        payload = json.dumps(
            config_payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def _input_hash(cls, path: Path) -> str:
        digest = hashlib.sha256()
        if path.is_dir():
            for child in sorted(item for item in path.rglob("*") if item.is_file()):
                relative_path = child.relative_to(path).as_posix()
                digest.update(relative_path.encode("utf-8"))
                digest.update(b"\0")
                digest.update(child.read_bytes())
                digest.update(b"\0")
            return digest.hexdigest()
        digest.update(path.read_bytes())
        return digest.hexdigest()

    @staticmethod
    def _git_metadata() -> tuple[str, Literal["clean", "dirty", "unknown"]]:
        cwd = Path.cwd()
        git_root = next((candidate for candidate in (cwd, *cwd.parents) if (candidate / ".git").exists()), None)
        if git_root is None:
            return "unknown", "unknown"
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=git_root,
                check=True,
                capture_output=True,
                text=True,
            )
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=git_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return "unknown", "unknown"
        commit_value = commit.stdout.strip() or "unknown"
        tree_state: Literal["clean", "dirty", "unknown"] = (
            "dirty" if status.stdout.strip() else "clean"
        )
        return commit_value, tree_state

    @staticmethod
    def _source_tree_hash() -> str:
        package_root = Path(__file__).resolve().parents[1]
        digest = hashlib.sha256()
        for path in sorted(package_root.rglob("*.py")):
            digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()


__all__ = [
    "BenchmarkHardGateSettings",
    "CandidateGraphResult",
    "BenchmarkRunMetadata",
    "EvaluationSettings",
    "GoldDatasetManifest",
    "GoldEvaluationSummary",
    "HardGateResult",
    "TechniqueEvaluationConfig",
    "TechniqueEvaluationReport",
    "TechniqueEvaluationRunner",
]
