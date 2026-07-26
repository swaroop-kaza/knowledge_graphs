"""YAML-driven configuration for product graph construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kg_construction.domain.modeling import ControlledVocabulary
from kg_construction.domain.schema import JsonType
from kg_construction.service_runtime import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
)
from kg_construction.structured.resolved_plan import IdentityScope
from kg_construction.structured.source_catalog import DeclaredColumnMapping

JsonDict = dict[str, Any]


class PipelineConfig(BaseModel):
    """Top-level pipeline execution settings."""

    model_config = ConfigDict(extra="forbid")

    name: str = "complete_data_to_kg"
    graph_model: Literal["hybrid"] = "hybrid"
    evidence_required: Literal[True] = True
    validation_required: Literal[True] = True
    validation_policy: Literal["reject"] = "reject"
    structured_mapping: Literal["semantic"] = "semantic"
    """The product mapping path.

    Direct table copying remains available as a research benchmark technique, but
    it is not a product-mode alternative to semantic, declaration-aware mapping.
    """
    schema_version: str = "v1"

    @field_validator("name", "schema_version")
    @classmethod
    def non_empty_identifier(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be empty")
        return stripped


class OllamaSettings(BaseModel):
    """Locked Ollama endpoint used by the mandatory Qwen3 provider."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["ollama"] = "ollama"
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    model: Literal["qwen3:1.7b-q4_K_M"] = DEFAULT_OLLAMA_MODEL
    temperature: float = Field(default=0.0, ge=0.0, le=0.0)
    structured_output: Literal[True] = True
    evidence_required: Literal[True] = True
    retry_on_invalid_output: Literal[True] = True
    max_retries: int = Field(default=3, ge=0)
    prompt_version: str = "evidence_first_claim_extraction_v2"

    @field_validator("base_url")
    @classmethod
    def valid_http_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return value.rstrip("/")

    @field_validator("prompt_version")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be empty")
        return stripped


class TechniqueConfig(BaseModel):
    """A mandatory capability assertion retained for configuration compatibility.

    These declarations do not select product runtime components: structured
    routing is source-driven and the unstructured ensemble is derived from the
    immutable model lock. A declaration is therefore an assertion that a
    capability is required, and cannot be used as an optional on/off switch.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: Literal[True] = True
    parameters: JsonDict = Field(default_factory=dict)


class UnstructuredConstructionSettings(BaseModel):
    """Fail-closed artifacts and gates for the platform-mandatory ensemble.

    Runtime selection and requested entity labels belong to
    :class:`ExtractionConfig`; keeping a second label list here previously made
    the complete YAML appear configured while the executable factory ignored
    that value.
    """

    model_config = ConfigDict(extra="forbid")

    models_lock_path: str = "models.lock.yaml"
    cache_root: str | None = None
    require_all_platform_providers: Literal[True] = True
    governed_schema_path: str | None = None
    held_out_gold_path: str | None = None
    require_governed_schema: Literal[True] = True
    require_pre_promotion_shacl: Literal[True] = True
    require_canonical_identity_ensemble: Literal[True] = True
    require_held_out_claim_evaluation: Literal[True] = True
    require_embedding_cross_encoder: Literal[True] = True
    require_external_linkers: Literal[True] = True
    nil_policy: Literal["retain_observed"] = "retain_observed"

    @field_validator("models_lock_path")
    @classmethod
    def non_empty_lock_path(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("models_lock_path must not be empty")
        if not stripped.endswith((".yaml", ".yml")):
            raise ValueError("models_lock_path must be a YAML lock file")
        return stripped

    @field_validator("cache_root")
    @classmethod
    def non_empty_cache_root(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("cache_root must be null or a non-empty path")
        return stripped

    @field_validator("governed_schema_path", "held_out_gold_path")
    @classmethod
    def non_empty_reviewed_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("reviewed artifact paths must be null or non-empty")
        return stripped

class StructuredForeignKeyConfig(BaseModel):
    """One declared foreign key, including composite join columns."""

    model_config = ConfigDict(extra="forbid")

    columns: list[str]
    target_table: str
    target_key: list[str] = Field(default_factory=list)
    relationship_name: str

    @field_validator("columns", "target_key")
    @classmethod
    def non_empty_column_names(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("foreign-key column names must not be empty")
        return values

    @model_validator(mode="after")
    def join_arity_matches(self) -> StructuredForeignKeyConfig:
        if not self.columns:
            raise ValueError("foreign key must declare at least one source column")
        if self.target_key and len(self.columns) != len(self.target_key):
            raise ValueError("foreign-key source and target column counts must match")
        if len(self.columns) != len(set(self.columns)):
            raise ValueError("foreign-key source columns must be unique")
        if self.target_key and len(self.target_key) != len(set(self.target_key)):
            raise ValueError("foreign-key target columns must be unique")
        if not self.target_table.strip():
            raise ValueError("foreign-key target_table must not be empty")
        if not self.relationship_name.strip():
            raise ValueError("foreign-key relationship_name must not be empty")
        return self


class StructuredTableConfig(BaseModel):
    """Authoritative structured-table identity and relationship declaration."""

    model_config = ConfigDict(extra="forbid")

    table_name: str
    record_type: str
    primary_key: list[str]
    alternate_keys: list[list[str]] = Field(default_factory=list)
    foreign_keys: list[StructuredForeignKeyConfig] = Field(default_factory=list)
    datatypes: dict[str, JsonType] = Field(default_factory=dict)
    column_mappings: dict[str, DeclaredColumnMapping] = Field(default_factory=dict)
    identity_scope: IdentityScope = IdentityScope.SOURCE
    controlled_vocabularies: list[ControlledVocabulary] = Field(default_factory=list)
    table_kind: Literal["auto", "entity", "association", "event", "edge"] = "auto"
    association_relationship_name: str | None = None

    @model_validator(mode="after")
    def primary_key_is_declared(self) -> StructuredTableConfig:
        if not self.table_name.strip():
            raise ValueError("structured table_name must not be empty")
        if not self.record_type.strip():
            raise ValueError("structured record_type must not be empty")
        if not self.primary_key or any(not item.strip() for item in self.primary_key):
            raise ValueError("a structured table must declare a non-empty primary key")
        if len(self.primary_key) != len(set(self.primary_key)):
            raise ValueError("structured table primary_key columns must be unique")
        if any(
            not key
            or any(not column.strip() for column in key)
            or len(key) != len(set(key))
            for key in self.alternate_keys
        ):
            raise ValueError(
                "structured table alternate keys must be non-empty and contain "
                "unique, non-empty columns"
            )
        key_signatures = [
            tuple(self.primary_key),
            *(tuple(key) for key in self.alternate_keys),
        ]
        if len(key_signatures) != len(set(key_signatures)):
            raise ValueError("structured table primary/alternate keys must be distinct")
        non_lexical_vocabulary_columns = sorted(
            column
            for column, mapping in self.column_mappings.items()
            if mapping.controlled_vocabulary
            and (
                mapping.datatype
                or self.datatypes.get(column)
                or "string"
            )
            != "string"
        )
        if non_lexical_vocabulary_columns:
            raise ValueError(
                "controlled-vocabulary codes must use string datatypes for "
                f"columns {non_lexical_vocabulary_columns}"
            )
        if self.table_kind == "edge" and not self.association_relationship_name:
            raise ValueError(
                "an edge table must declare association_relationship_name"
            )
        return self


class StructuredConstructionSettings(BaseModel):
    """Structured-pillar declarations used by API, CLI, and certification."""

    model_config = ConfigDict(extra="forbid")

    require_declarations: bool = False
    tables: list[StructuredTableConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def table_names_are_unique(self) -> StructuredConstructionSettings:
        names = [table.table_name for table in self.tables]
        if len(names) != len(set(names)):
            raise ValueError("structured table declarations must have unique table_name values")
        if self.require_declarations and not self.tables:
            raise ValueError(
                "structured.tables must not be empty when require_declarations=true"
            )
        if self.require_declarations:
            for table in self.tables:
                vocabularies = {
                    vocabulary.name
                    for vocabulary in table.controlled_vocabularies
                }
                bindings = {
                    mapping.controlled_vocabulary
                    for mapping in table.column_mappings.values()
                    if mapping.controlled_vocabulary
                }
                unbound = sorted(vocabularies - bindings)
                if unbound:
                    raise ValueError(
                        f"structured table {table.table_name!r} must explicitly "
                        "bind controlled vocabularies through column_mappings: "
                        f"{unbound}"
                    )
        return self


class StructuredLoggingSettings(BaseModel):
    """Production structured logging settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    redact_pii: bool = True


class RunManifestSettings(BaseModel):
    """Run manifest persistence settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    output_dir: str = "data/processed/manifests"


class IncrementalSyncSettings(BaseModel):
    """Incremental source processing settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    watermark_path: str = "data/processed/watermarks.json"
    watermark_field: str = "updated_at"


class ErrorQuarantineSettings(BaseModel):
    """Invalid source/candidate quarantine settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    path: str = "data/processed/quarantine.jsonl"


class HumanReviewQueueSettings(BaseModel):
    """Human review queue persistence settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    path: str = "data/processed/human_review_queue.json"


class PIIRedactionSettings(BaseModel):
    """Policy flags for deterministic PII redaction."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    replacement: str = "[REDACTED]"
    redact_emails: bool = True
    redact_phone_numbers: bool = True
    redact_ssn: bool = True
    redact_ip_addresses: bool = True
    redact_api_keys: bool = True


class ProductionSettings(BaseModel):
    """Production hardening settings around the construction workflow."""

    model_config = ConfigDict(extra="forbid")

    structured_logging: StructuredLoggingSettings = Field(default_factory=StructuredLoggingSettings)
    run_manifest: RunManifestSettings = Field(default_factory=RunManifestSettings)
    incremental_sync: IncrementalSyncSettings = Field(default_factory=IncrementalSyncSettings)
    error_quarantine: ErrorQuarantineSettings = Field(default_factory=ErrorQuarantineSettings)
    human_review_queue: HumanReviewQueueSettings = Field(default_factory=HumanReviewQueueSettings)
    pii_redaction: PIIRedactionSettings = Field(default_factory=PIIRedactionSettings)
    continue_on_error: bool = False

    @model_validator(mode="after")
    def validate_enabled_paths(self) -> ProductionSettings:
        path_settings = {
            "run_manifest.output_dir": (
                self.run_manifest.enabled,
                self.run_manifest.output_dir,
            ),
            "incremental_sync.watermark_path": (
                self.incremental_sync.enabled,
                self.incremental_sync.watermark_path,
            ),
            "error_quarantine.path": (
                self.error_quarantine.enabled,
                self.error_quarantine.path,
            ),
            "human_review_queue.path": (
                self.human_review_queue.enabled,
                self.human_review_queue.path,
            ),
        }
        for name, (enabled, path) in path_settings.items():
            if enabled and not path.strip():
                raise ValueError(f"{name} must be set when enabled")
        return self


ALLOWED_TECHNIQUES: dict[str, set[str]] = {
    "loaders": {
        "api_response",
        "chat_transcript",
        "csv",
        "email",
        "event_payload",
        "excel",
        "html",
        "json",
        "jsonld",
        "logs",
        "markdown",
        "pdf",
        "report",
        "scientific_paper",
        "sql",
        "structured_api",
        "text",
        "webpage",
        "xml",
    },
    "mappers": {
        "csvw",
        "direct_table",
        "event_graph",
        "html_metadata",
        "jsonld",
        "jsonpath",
        "log_event",
        "obda_virtual",
        "r2rml",
        "rml",
        "semantic_catalog",
    },
    "extractors": {
        "evidence_first_unstructured",
        "gazetteer",
        "gliner",
        "gliner_relex",
        "mlx_qwen3",
        "ollama_graph",
        "ollama_qwen3",
        "openie",
        "regex",
        "rebel",
        "spacy",
        "supervised_re",
        "transformer_ner",
        "transformers_qwen3",
    },
    "resolution": {
        "alias_table",
        "cross_encoder",
        "dedupe",
        "embedding",
        "exact",
        "fuzzy",
        "graph_based",
        "human_review",
        "rapidfuzz",
        "splink",
    },
    "validation": {
        "confidence_threshold",
        "evidence_required",
        "json_schema",
        "property_graph",
        "referential_integrity",
        "schema_registry",
        "shacl",
    },
    "graph_store": {
        "evidence_graph",
        "ladybug",
        "memgraph",
        "neo4j",
        "networkx",
        "property_graph",
        "rdf",
        "rdflib",
    },
    "enrichment": {
        "community_detection",
        "external_kg_linking",
        "graph_analytics",
        "node_embeddings",
        "pagerank",
    },
    "evaluation": {
        "benchmark",
        "benchmark_runner",
        "construction_metrics",
        "intended_use_metrics",
        "knowledge_graph_quality",
        "metrics",
        "recommendation",
        "schema_quality",
        "validation_failure_metrics",
    },
}


_DEFAULT_ENTITY_LABELS: tuple[str, ...] = (
    "Person",
    "Organization",
    "Location",
    "Event",
    "Product",
    "Date",
    "Concept",
    "System",
    "Service",
    "Incident",
)


class ExtractionConfig(BaseModel):
    """Select the deterministic or locked model-backed extraction profile.

    Individual providers are deliberately not runtime switches. The immutable
    model lock selects every platform-applicable provider (GLiNER, GLiNER RelEx,
    Transformers, spaCy, Ollama, and MLX on Apple Silicon), so a run cannot omit
    one provider or substitute an untracked model.
    """

    model_config = ConfigDict(extra="forbid")

    runtime_profile: Literal["deterministic", "model_backed"] = "deterministic"
    entity_labels: tuple[str, ...] = _DEFAULT_ENTITY_LABELS

    @field_validator("entity_labels")
    @classmethod
    def _unique_entity_labels(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if not normalized or any(not value for value in normalized):
            raise ValueError("entity_labels must contain non-empty labels")
        if len({value.casefold() for value in normalized}) != len(normalized):
            raise ValueError("entity_labels must be unique ignoring case")
        return normalized


class QualityThresholdOverride(BaseModel):
    """Override one formation/quality gate threshold by ``(family, metric)``."""

    model_config = ConfigDict(extra="forbid")

    family: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    operator: Literal[">=", "<="]
    threshold: float


class EvaluationThresholdsConfig(BaseModel):
    """Configurable evaluation thresholds.

    ``convert`` reports these gates without authorizing publication. Every field
    maps directly to an executed quality-gate metric; comparative ranking has a
    separate configuration under :mod:`kg_construction.research`.
    """

    model_config = ConfigDict(extra="forbid")

    # supervised-quality floors (quality-gate families)
    ner_f1_min: float = Field(default=0.80, ge=0.0, le=1.0)
    span_f1_min: float = Field(default=0.80, ge=0.0, le=1.0)
    relation_f1_min: float = Field(default=0.80, ge=0.0, le=1.0)
    coreference_f1_min: float = Field(default=0.80, ge=0.0, le=1.0)
    linking_accuracy_min: float = Field(default=0.80, ge=0.0, le=1.0)
    query_accuracy_min: float = Field(default=0.80, ge=0.0, le=1.0)
    vector_recall_at_k_min: float = Field(default=0.70, ge=0.0, le=1.0)
    max_retry_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    quality_overrides: tuple[QualityThresholdOverride, ...] = ()

    def floor_overrides(self) -> dict[tuple[str, str], float]:
        """Return ``(family, metric) -> threshold`` for the named floor knobs."""

        return {
            ("ner", "entity_f1"): self.ner_f1_min,
            ("ner", "span_f1"): self.span_f1_min,
            ("ner", "entity_gold_coverage"): self.ner_f1_min,
            ("ner", "span_gold_coverage"): self.span_f1_min,
            ("relation", "relation_f1"): self.relation_f1_min,
            ("relation", "relation_gold_coverage"): self.relation_f1_min,
            ("coreference", "coreference_b3_f1"): self.coreference_f1_min,
            ("linking", "entity_linking_accuracy"): self.linking_accuracy_min,
            ("query", "competency_question_accuracy"): self.query_accuracy_min,
            ("vector", "vector_recall_at_k"): self.vector_recall_at_k_min,
            ("operational", "retry_rate"): self.max_retry_rate,
        }


class DataToKGConfig(BaseModel):
    """Complete config file model."""

    model_config = ConfigDict(extra="forbid")

    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    structured: StructuredConstructionSettings = Field(
        default_factory=StructuredConstructionSettings
    )
    unstructured: UnstructuredConstructionSettings = Field(
        default_factory=UnstructuredConstructionSettings
    )
    llm: OllamaSettings = Field(default_factory=OllamaSettings)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    evaluation_thresholds: EvaluationThresholdsConfig = Field(
        default_factory=EvaluationThresholdsConfig
    )
    loaders: dict[str, TechniqueConfig] = Field(default_factory=dict)
    mappers: dict[str, TechniqueConfig] = Field(default_factory=dict)
    extractors: dict[str, TechniqueConfig] = Field(default_factory=dict)
    resolution: dict[str, TechniqueConfig] = Field(default_factory=dict)
    validation: dict[str, TechniqueConfig] = Field(default_factory=dict)
    graph_store: dict[str, TechniqueConfig] = Field(default_factory=dict)
    enrichment: dict[str, TechniqueConfig] = Field(default_factory=dict)
    evaluation: dict[str, TechniqueConfig] = Field(default_factory=dict)
    evaluation_suite_path: str | None = None
    production: ProductionSettings = Field(default_factory=ProductionSettings)

    @field_validator("evaluation_suite_path")
    @classmethod
    def non_empty_evaluation_suite_path(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("evaluation_suite_path must be null or a non-empty path")
        if not stripped.endswith((".yaml", ".yml", ".json")):
            raise ValueError(
                "evaluation_suite_path must reference a YAML or JSON fixture"
            )
        return stripped

    @classmethod
    def from_yaml(cls, path: str | Path) -> DataToKGConfig:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(payload)

    @model_validator(mode="after")
    def validate_technique_names_and_dependencies(self) -> DataToKGConfig:
        groups = {
            "loaders": self.loaders,
            "mappers": self.mappers,
            "extractors": self.extractors,
            "resolution": self.resolution,
            "validation": self.validation,
            "graph_store": self.graph_store,
            "enrichment": self.enrichment,
            "evaluation": self.evaluation,
        }
        for group_name, configured in groups.items():
            unknown = set(configured) - ALLOWED_TECHNIQUES[group_name]
            if unknown:
                raise ValueError(f"Unknown {group_name} technique(s): {sorted(unknown)}")
        if "ollama_graph" in self.extractors and self.llm.provider != "ollama":
            raise ValueError("ollama_graph extractor requires llm.provider='ollama'")
        if self.pipeline.evidence_required and not self.validation.get(
            "evidence_required",
            TechniqueConfig(enabled=True),
        ).enabled:
            raise ValueError("pipeline.evidence_required requires validation.evidence_required")
        return self

    def enabled_techniques(self) -> dict[str, list[str]]:
        groups = {
            "loaders": self.loaders,
            "mappers": self.mappers,
            "extractors": self.extractors,
            "resolution": self.resolution,
            "validation": self.validation,
            "graph_store": self.graph_store,
            "enrichment": self.enrichment,
            "evaluation": self.evaluation,
        }
        return {
            group: sorted(name for name, item in techniques.items() if item.enabled)
            for group, techniques in groups.items()
        }


__all__ = [
    "DataToKGConfig",
    "ErrorQuarantineSettings",
    "EvaluationThresholdsConfig",
    "ExtractionConfig",
    "QualityThresholdOverride",
    "HumanReviewQueueSettings",
    "IncrementalSyncSettings",
    "OllamaSettings",
    "PIIRedactionSettings",
    "PipelineConfig",
    "ProductionSettings",
    "RunManifestSettings",
    "StructuredConstructionSettings",
    "StructuredForeignKeyConfig",
    "StructuredLoggingSettings",
    "StructuredTableConfig",
    "TechniqueConfig",
    "UnstructuredConstructionSettings",
]
