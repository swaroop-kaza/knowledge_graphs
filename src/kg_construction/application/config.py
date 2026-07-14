"""YAML-driven configuration for the Data-to-KG framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

JsonDict = dict[str, Any]


class PipelineConfig(BaseModel):
    """Top-level pipeline execution settings."""

    model_config = ConfigDict(extra="forbid")

    name: str = "complete_data_to_kg"
    graph_model: Literal["property", "rdf", "hybrid"] = "hybrid"
    evidence_required: bool = True
    validation_required: bool = True
    validation_policy: Literal["reject", "quarantine", "repair", "send_to_human_review"] = "reject"
    schema_version: str = "v1"

    @field_validator("name", "schema_version")
    @classmethod
    def non_empty_identifier(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be empty")
        return stripped


class OllamaSettings(BaseModel):
    """LLM provider configuration; Ollama is the default."""

    model_config = ConfigDict(extra="forbid")

    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "llama3.1"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    structured_output: bool = True
    evidence_required: bool = True
    retry_on_invalid_output: bool = True
    max_retries: int = Field(default=3, ge=0)
    prompt_version: str = "ollama_graph_extraction_v1"

    @field_validator("base_url")
    @classmethod
    def valid_http_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return value.rstrip("/")

    @field_validator("model", "prompt_version", "provider")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be empty")
        return stripped


class TechniqueConfig(BaseModel):
    """Generic on/off switch and parameter bag for a technique."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    parameters: JsonDict = Field(default_factory=dict)


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
        "excel",
        "json",
        "logs",
        "markdown",
        "pdf",
        "report",
        "sql",
        "text",
        "webpage",
        "xml",
    },
    "mappers": {
        "csvw",
        "direct_table",
        "jsonld",
        "jsonpath",
        "obda_virtual",
        "r2rml",
        "rml",
    },
    "extractors": {
        "gazetteer",
        "gliner",
        "ollama_graph",
        "openie",
        "regex",
        "rebel",
        "spacy",
        "supervised_re",
        "transformer_ner",
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
        "schema_registry",
        "shacl",
    },
    "graph_store": {"kuzu", "memgraph", "neo4j", "networkx", "rdf", "rdflib"},
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
        "metrics",
        "recommendation",
    },
}


class DataToKGConfig(BaseModel):
    """Complete config file model."""

    model_config = ConfigDict(extra="forbid")

    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    llm: OllamaSettings = Field(default_factory=OllamaSettings)
    loaders: dict[str, TechniqueConfig] = Field(default_factory=dict)
    mappers: dict[str, TechniqueConfig] = Field(default_factory=dict)
    extractors: dict[str, TechniqueConfig] = Field(default_factory=dict)
    resolution: dict[str, TechniqueConfig] = Field(default_factory=dict)
    validation: dict[str, TechniqueConfig] = Field(default_factory=dict)
    graph_store: dict[str, TechniqueConfig] = Field(default_factory=dict)
    enrichment: dict[str, TechniqueConfig] = Field(default_factory=dict)
    evaluation: dict[str, TechniqueConfig] = Field(default_factory=dict)
    production: ProductionSettings = Field(default_factory=ProductionSettings)

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
        if self.extractors.get("ollama_graph", TechniqueConfig()).enabled and self.llm.provider != "ollama":
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
    "HumanReviewQueueSettings",
    "IncrementalSyncSettings",
    "OllamaSettings",
    "PIIRedactionSettings",
    "PipelineConfig",
    "ProductionSettings",
    "RunManifestSettings",
    "StructuredLoggingSettings",
    "TechniqueConfig",
]
