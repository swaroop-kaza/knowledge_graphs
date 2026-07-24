"""Validate public configuration examples.

The example YAML files are part of the public API. This checker keeps them
parseable, model-backed where possible, and aligned with generated-artifact
policy without calling external services.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from kg_construction.application.benchmark_runner import TechniqueEvaluationConfig
from kg_construction.application.config import DataToKGConfig
from kg_construction.application.graph_ml import graph_ml_workflow_config_from_dict
from kg_construction.application.hybrid_gliner import HybridGlinerPipelineConfig
from kg_construction.infrastructure.evaluation import VersionedGraphEvaluationSuite
from kg_construction.infrastructure.extractors.relations import RelationTypingConfig
from kg_construction.infrastructure.graph_stores.ingestion import (
    LadybugIngestionConfig,
    Neo4jIngestionConfig,
)
from kg_construction.infrastructure.ontology.unstructured_governance import (
    GovernedUnstructuredSchema,
)
from kg_construction.infrastructure.production.retry import RetryPolicyConfig

JsonDict = dict[str, Any]
CONFIG_DIR = Path("configs")


class DeadLetterConfig(BaseModel):
    """Dead-letter queue config subset used by production ingestion examples."""

    model_config = ConfigDict(extra="forbid")

    path: str


class ProductionReplayConfig(BaseModel):
    """Replay command examples for production ingestion docs."""

    model_config = ConfigDict(extra="forbid")

    command_examples: list[str] = Field(default_factory=list)


class GraphStoreBatchBenchmarkConfig(BaseModel):
    """Config subset for graph-store batch benchmark examples."""

    model_config = ConfigDict(extra="forbid")

    node_counts: list[int]
    edge_factor: int = Field(gt=0)
    batch_sizes: list[int]
    output: str


EXPECTED_CONFIGS = {
    "data_to_kg_complete.example.yaml",
    "evaluation.example.yaml",
    "gliner_ollama_relation_typing.example.yaml",
    "governed_unstructured_ontology.yaml",
    "graph_evaluation_suite.sample_csv.yaml",
    "graph_evaluation_suite.sample_text.yaml",
    "graph_ml.example.yaml",
    "integration_profiles.example.yaml",
    "ollama_gliner_label_proposal.example.yaml",
    "production_ingestion.example.yaml",
}


def main() -> int:
    failures: list[str] = []
    loaded: dict[str, JsonDict] = {}

    actual = {path.name for path in CONFIG_DIR.glob("*.yaml")}
    missing = EXPECTED_CONFIGS - actual
    extra = actual - EXPECTED_CONFIGS
    if missing:
        failures.append(f"Missing expected config file(s): {sorted(missing)}")
    if extra:
        failures.append(f"Unexpected config file(s): {sorted(extra)}")

    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        try:
            payload = _load_yaml(path)
            loaded[path.name] = payload
            _validate_known_config(path.name, payload)
            _validate_generated_paths(path.name, payload)
        except Exception as exc:  # noqa: BLE001 - config checker should report all failures uniformly.
            failures.append(f"{path}: {type(exc).__name__}: {exc}")

    if failures:
        for failure in failures:
            print(failure)
        return 1

    print(f"config-check-ok: {len(loaded)} config files checked")
    return 0


def _load_yaml(path: Path) -> JsonDict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("top-level YAML document must be a mapping")
    return payload


def _validate_known_config(filename: str, payload: JsonDict) -> None:
    if filename == "data_to_kg_complete.example.yaml":
        DataToKGConfig.model_validate(payload)
        _require_sections(
            filename,
            payload,
            "pipeline",
            "llm",
            "loaders",
            "mappers",
            "extractors",
            "resolution",
            "validation",
            "graph_store",
            "enrichment",
            "evaluation",
            "production",
        )
    elif filename == "evaluation.example.yaml":
        TechniqueEvaluationConfig.model_validate(payload)
    elif filename == "ollama_gliner_label_proposal.example.yaml":
        HybridGlinerPipelineConfig.model_validate(payload)
        _require_sections(filename, payload, "llm", "label_proposal", "review", "gliner")
    elif filename == "gliner_ollama_relation_typing.example.yaml":
        RelationTypingConfig.model_validate(payload.get("relation_typing", payload))
    elif filename == "graph_ml.example.yaml":
        graph_ml_workflow_config_from_dict(payload)
    elif filename == "production_ingestion.example.yaml":
        _validate_production_ingestion(payload)
    elif filename == "integration_profiles.example.yaml":
        _validate_integration_profiles(payload)
    elif filename == "governed_unstructured_ontology.yaml":
        GovernedUnstructuredSchema.from_yaml(CONFIG_DIR / filename)
    elif filename.startswith("graph_evaluation_suite."):
        VersionedGraphEvaluationSuite.model_validate(payload)
    else:
        raise ValueError(f"no checker registered for {filename}")


def _validate_production_ingestion(payload: JsonDict) -> None:
    root = _mapping(payload, "production_ingestion")
    _require_sections(
        "production_ingestion.example.yaml",
        root,
        "retry",
        "dead_letter",
        "neo4j",
        "ladybug",
        "replay",
    )

    RetryPolicyConfig(**{**_mapping(root, "retry"), "sleep": False})
    DeadLetterConfig.model_validate(_mapping(root, "dead_letter"))
    LadybugIngestionConfig(
        **_dataclass_kwargs(LadybugIngestionConfig, _mapping(root, "ladybug"))
    )
    _validate_neo4j_ingestion_config(_mapping(root, "neo4j"))
    replay = ProductionReplayConfig.model_validate(_mapping(root, "replay"))
    if not replay.command_examples:
        raise ValueError("production_ingestion.replay.command_examples must not be empty")


def _validate_neo4j_ingestion_config(payload: JsonDict) -> None:
    allowed_extra = {"password_env"}
    unknown = set(payload) - _dataclass_field_names(Neo4jIngestionConfig) - allowed_extra
    if unknown:
        raise ValueError(f"unknown neo4j ingestion key(s): {sorted(unknown)}")
    if "password_env" in payload and not str(payload["password_env"]).strip():
        raise ValueError("neo4j.password_env must not be empty when provided")
    constructor_payload = {key: value for key, value in payload.items() if key != "password_env"}
    Neo4jIngestionConfig(**_dataclass_kwargs(Neo4jIngestionConfig, constructor_payload))


def _validate_integration_profiles(payload: JsonDict) -> None:
    root = _mapping(payload, "integration_profiles")
    _require_sections(
        "integration_profiles.example.yaml",
        root,
        "neo4j_live",
        "ladybug_live",
        "graph_store_batch_benchmark",
    )
    neo4j = _mapping(root, "neo4j_live")
    for key in ("enabled_env", "uri_env", "user_env", "password_env", "tests"):
        _require_key("integration_profiles.neo4j_live", neo4j, key)
    if not isinstance(neo4j["tests"], list) or not neo4j["tests"]:
        raise ValueError("integration_profiles.neo4j_live.tests must be a non-empty list")

    ladybug = _mapping(root, "ladybug_live")
    for key in ("enabled_env", "database_path", "tests"):
        _require_key("integration_profiles.ladybug_live", ladybug, key)
    if not str(ladybug["database_path"]).startswith("data/processed/"):
        raise ValueError(
            "integration_profiles.ladybug_live.database_path must be under data/processed/"
        )
    if not isinstance(ladybug["tests"], list) or not ladybug["tests"]:
        raise ValueError(
            "integration_profiles.ladybug_live.tests must be a non-empty list"
        )

    benchmark = GraphStoreBatchBenchmarkConfig.model_validate(
        _mapping(root, "graph_store_batch_benchmark")
    )
    if any(count <= 0 for count in benchmark.node_counts):
        raise ValueError("graph_store_batch_benchmark.node_counts must be positive")
    if any(size <= 0 for size in benchmark.batch_sizes):
        raise ValueError("graph_store_batch_benchmark.batch_sizes must be positive")
    if not benchmark.output.startswith("reports/"):
        raise ValueError("graph_store_batch_benchmark.output must be under reports/")


def _validate_generated_paths(filename: str, payload: Any) -> None:
    for dotted_path, value in _walk(payload):
        if not isinstance(value, str):
            continue
        key = dotted_path.rsplit(".", maxsplit=1)[-1]
        if key in {
            "evaluation_suite_path",
            "governed_schema_path",
            "held_out_gold_path",
            "models_lock_path",
        }:
            continue
        if key not in {"path", "output", "output_dir", "database_path", "watermark_path"}:
            continue
        if value.startswith(("http://", "https://", "bolt://", "neo4j://", "memory")):
            continue
        if not value.startswith(("data/processed/", "reports/")):
            raise ValueError(
                f"{filename}: generated path {dotted_path}={value!r} must be under "
                "data/processed/ or reports/"
            )


def _walk(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        rows: list[tuple[str, Any]] = []
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_walk(item, child_prefix))
        return rows
    if isinstance(value, list):
        rows = []
        for index, item in enumerate(value):
            rows.extend(_walk(item, f"{prefix}[{index}]"))
        return rows
    return [(prefix, value)]


def _mapping(payload: JsonDict, key: str) -> JsonDict:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _require_sections(filename: str, payload: JsonDict, *keys: str) -> None:
    for key in keys:
        _require_key(filename, payload, key)


def _require_key(context: str, payload: JsonDict, key: str) -> None:
    if key not in payload:
        raise ValueError(f"{context} missing required key {key!r}")


def _dataclass_field_names(model: type[Any]) -> set[str]:
    if not is_dataclass(model):
        raise TypeError(f"{model} is not a dataclass")
    return {item.name for item in fields(model)}


def _dataclass_kwargs(model: type[Any], payload: JsonDict) -> JsonDict:
    allowed = _dataclass_field_names(model)
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown {model.__name__} key(s): {sorted(unknown)}")
    return {key: value for key, value in payload.items() if key in allowed}


if __name__ == "__main__":
    raise SystemExit(main())
