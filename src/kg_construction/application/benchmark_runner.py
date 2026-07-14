"""End-to-end technique evaluation and selection runner."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from kg_construction.application.dataset_profiler import DatasetProfile, DatasetProfiler
from kg_construction.application.recommendation_engine import (
    RecommendationEngine,
    RecommendationSummary,
)
from kg_construction.application.registry import (
    PluginRegistry,
    TechniqueDescriptor,
    default_registry,
)
from kg_construction.application.scoring import (
    PenaltyWeights,
    ProductionThresholds,
    ScoringEngine,
    ScoringWeights,
    TechniqueScore,
)
from kg_construction.application.technique_selector import ApplicabilityResult, TechniqueSelector
from kg_construction.application.workflow import ConstructionResult, ConstructionWorkflow
from kg_construction.domain.graph_objects import PropertyGraph, RDFGraph
from kg_construction.domain.models import ConstructionBatch, SourceRecord
from kg_construction.infrastructure.evaluation.accuracy_metrics import AccuracyMetrics
from kg_construction.infrastructure.evaluation.coverage_metrics import CoverageMetrics
from kg_construction.infrastructure.evaluation.entity_resolution_metrics import (
    EntityResolutionMetrics,
)
from kg_construction.infrastructure.evaluation.operational_metrics import (
    OperationalMetrics,
    OperationalMetricsRecorder,
    RuntimeReport,
)
from kg_construction.infrastructure.evaluation.retrieval_metrics import RetrievalMetrics
from kg_construction.infrastructure.evaluation.statistical_confidence import wilson_score_interval
from kg_construction.infrastructure.evaluation.validation_metrics import ValidationMetrics
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

DEFAULT_GAZETTEER = {
    "Organization": ["Platform Team", "Research", "Compiler", "Flight", "ML"],
    "Incident": ["INCIDENT-1042"],
    "System": ["Recommendation API", "Feature Store"],
}

TIE_BREAKING_POLICY = [
    "final_score_desc",
    "production_thresholds_pass_desc",
    "latency_asc",
    "maintainability_desc",
    "technique_name_asc",
]


class EvaluationSettings(BaseModel):
    """Settings controlling which techniques are benchmarked."""

    model_config = ConfigDict(extra="forbid")

    run_all_applicable_techniques: bool = True
    force_all_techniques: bool = False
    gold_dataset_path: str | None = None
    human_review_sample_size: int = 100
    bootstrap_iterations: int = 1000
    allow_external_services: bool = False
    deterministic_runtime: bool = True
    fixed_latency_seconds: float = 0.0
    fixed_throughput: float = 1000.0
    fixed_memory_peak_mb: float = 0.0
    fixed_graph_write_time_seconds: float = 0.0
    fixed_ollama_prompt_tokens: int = 0
    fixed_ollama_output_tokens: int = 0
    fixed_ollama_cost_proxy: float = 0.0


class TechniqueEvaluationConfig(BaseModel):
    """YAML config for benchmark and recommendation runs."""

    model_config = ConfigDict(extra="forbid")

    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    applicability_threshold: float = 0.40
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    penalties: PenaltyWeights = Field(default_factory=PenaltyWeights)
    production_thresholds: ProductionThresholds = Field(default_factory=ProductionThresholds)
    llm: dict[str, Any] = Field(default_factory=lambda: {"provider": "ollama", "model": "llama3.1"})

    @classmethod
    def from_yaml(cls, path: str | Path) -> TechniqueEvaluationConfig:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(payload)


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
    error: str | None = None


class BenchmarkRunMetadata(BaseModel):
    """Reproducibility metadata for one benchmark execution."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    timestamp: str
    config_hash: str
    git_commit: str
    input_hash: str
    source_type: str | None = None
    techniques: str
    deterministic_runtime: bool
    tie_breaking_policy: list[str] = Field(default_factory=list)
    model_settings: dict[str, Any] = Field(default_factory=dict)


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
    def best_technique(self) -> str | None:
        return self.recommendations.best_single_technique.technique_name


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
        self.recommender = RecommendationEngine(self.config.production_thresholds)

    def run(
        self,
        *,
        input_path: str | Path,
        source_type: str | None = None,
        techniques: str = "applicable",
        output_dir: str | Path | None = None,
    ) -> TechniqueEvaluationReport:
        generated_at = datetime.now(UTC).isoformat()
        sources = self._load_sources(input_path, source_type)
        profile = self.profiler.profile_sources(sources, input_path=str(input_path))
        force_all = techniques == "all" or self.config.evaluation.force_all_techniques
        applicability = self.selector.select(profile, force_all=force_all)
        selected_names = self._selected_names(applicability, techniques)
        candidate_specs = self._candidate_specs(sources, selected_names, profile)
        candidates = [
            self._run_candidate(spec, sources, profile)
            for spec in candidate_specs
        ]
        ranked_scores = self._score_candidates(candidates)
        recommendations = self.recommender.recommend(ranked_scores)
        report = TechniqueEvaluationReport(
            generated_at=generated_at,
            run_metadata=self._build_run_metadata(
                input_path=Path(input_path),
                source_type=source_type,
                techniques=techniques,
                timestamp=generated_at,
            ),
            input_path=str(input_path),
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
    ) -> CandidateGraphResult:
        processed_items = max(1, profile.record_count or profile.source_count)
        result: ConstructionResult | None = None
        error: str | None = None
        with OperationalMetricsRecorder(processed_items=processed_items) as recorder:
            try:
                workflow = ConstructionWorkflow(
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
        metrics = self._evaluate_result(result, runtime_report, profile)
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
    ) -> dict[str, float]:
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
        metrics.update(EntityResolutionMetrics().evaluate(result.batch))
        metrics.update(RetrievalMetrics().evaluate())
        metrics.update(OperationalMetrics().evaluate(runtime_report))
        metrics["estimated_precision"] = (
            metrics["evidence_coverage"]
            * metrics["schema_validity_score"]
            * (1.0 - metrics["unsupported_extraction_rate"])
        )
        metrics["entity_count"] = float(len(result.batch.entities))
        metrics["relation_count"] = float(len(result.batch.relations))
        return metrics

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
            total = max(1, int(candidate.evaluation_report.get("entity_count", 0) + candidate.evaluation_report.get("relation_count", 0)))
            valid_estimate = int(candidate.evaluation_report.get("estimated_precision", 0.0) * total)
            score = score.model_copy(update={"confidence_interval": wilson_score_interval(valid_estimate, total)})
            scores.append(score)
        return sorted(scores, key=self._ranking_key)

    def _ranking_key(self, score: TechniqueScore) -> tuple[float, bool, float, float, str]:
        return (
            -round(score.final_score, 12),
            not self._passes_production_thresholds(score),
            round(score.metrics.get("latency", 0.0), 12),
            -round(score.dimensions.get("maintainability", 0.0), 12),
            score.technique_name,
        )

    def _passes_production_thresholds(self, score: TechniqueScore) -> bool:
        metrics = score.metrics
        unsupported_extraction_rate = metrics.get(
            "unsupported_extraction_rate",
            metrics.get("hallucination_risk_proxy", metrics.get("hallucination_rate", 1.0)),
        )
        thresholds = self.config.production_thresholds
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

    def _build_run_metadata(
        self,
        *,
        input_path: Path,
        source_type: str | None,
        techniques: str,
        timestamp: str,
    ) -> BenchmarkRunMetadata:
        config_hash = self._config_hash()
        input_hash = self._input_hash(input_path)
        run_seed = json.dumps(
            {
                "timestamp": timestamp,
                "config_hash": config_hash,
                "input_hash": input_hash,
                "techniques": techniques,
            },
            sort_keys=True,
        )
        run_id = hashlib.sha256(run_seed.encode("utf-8")).hexdigest()[:16]
        return BenchmarkRunMetadata(
            run_id=run_id,
            timestamp=timestamp,
            config_hash=config_hash,
            git_commit=self._git_commit(),
            input_hash=input_hash,
            source_type=source_type,
            techniques=techniques,
            deterministic_runtime=self.config.evaluation.deterministic_runtime,
            tie_breaking_policy=TIE_BREAKING_POLICY,
            model_settings={
                "llm": self.config.llm,
                "allow_external_services": self.config.evaluation.allow_external_services,
            },
        )

    def _config_hash(self) -> str:
        payload = json.dumps(
            self.config.model_dump(mode="json"),
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
    def _git_commit() -> str:
        cwd = Path.cwd()
        git_root = next((candidate for candidate in (cwd, *cwd.parents) if (candidate / ".git").exists()), None)
        if git_root is None:
            return "unknown"
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=git_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return "unknown"
        return completed.stdout.strip() or "unknown"


__all__ = [
    "CandidateGraphResult",
    "BenchmarkRunMetadata",
    "EvaluationSettings",
    "TechniqueEvaluationConfig",
    "TechniqueEvaluationReport",
    "TechniqueEvaluationRunner",
]
