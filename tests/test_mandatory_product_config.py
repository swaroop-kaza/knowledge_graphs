"""Contracts for the fail-closed, three-pillar product configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from kg_construction import load_sources
from kg_construction.construction.config import DataToKGConfig
from kg_construction.research.benchmark_runner import TechniqueEvaluationConfig
from kg_construction.research.workflow import (
    draft_workflow_for_source_type,
)

_COMPLETE_CONFIG = Path("configs/data_to_kg_complete.example.yaml")
_EVALUATION_CONFIG = Path("configs/evaluation.example.yaml")


def test_complete_profile_has_non_disableable_product_gates() -> None:
    config = DataToKGConfig.from_yaml(_COMPLETE_CONFIG)

    assert config.pipeline.graph_model == "hybrid"
    assert config.pipeline.structured_mapping == "semantic"
    assert config.pipeline.evidence_required is True
    assert config.pipeline.validation_required is True
    assert config.pipeline.validation_policy == "reject"
    with pytest.raises(ValidationError):
        DataToKGConfig.model_validate(
            {"pipeline": {"evidence_required": False}}
        )
    with pytest.raises(ValidationError):
        DataToKGConfig.model_validate(
            {"pipeline": {"structured_mapping": "direct"}}
        )


def test_complete_profile_declares_identity_relationship_and_vocabulary() -> None:
    config = DataToKGConfig.from_yaml(_COMPLETE_CONFIG)

    assert config.structured.require_declarations is True
    assert len(config.structured.tables) == 1
    table = config.structured.tables[0]
    assert table.table_name == "examples/data_to_kg/sample.csv"
    assert table.primary_key == ["id"]
    assert table.foreign_keys[0].columns == ["manager_id"]
    assert table.foreign_keys[0].target_key == ["id"]
    assert table.foreign_keys[0].relationship_name == "reports_to"
    assert table.controlled_vocabularies[0].name == "Team"
    assert table.controlled_vocabularies[0].closed is True


def test_complete_profile_constructs_declared_edges_and_deduplicated_concepts() -> None:
    config = DataToKGConfig.from_yaml(_COMPLETE_CONFIG)
    sources = load_sources(
        "examples/data_to_kg/sample.csv",
        source_type="csv",
    )
    result = draft_workflow_for_source_type("csv", config).run(sources)

    assert len(
        [node for node in result.property_graph.nodes if "Team" in node.labels]
    ) == 3
    assert len(
        [edge for edge in result.property_graph.edges if edge.type == "reports_to"]
    ) == 2
    assert result.validation_report.valid


def test_complete_profile_mirrors_the_locked_platform_model_ensemble() -> None:
    config = DataToKGConfig.from_yaml(_COMPLETE_CONFIG)

    assert config.extraction.runtime_profile == "model_backed"
    assert set(config.extraction.entity_labels) <= {
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
    }
    assert config.unstructured.require_all_platform_providers is True
    assert config.unstructured.models_lock_path == "models.lock.yaml"
    assert set(config.extractors) == {
        "evidence_first_unstructured",
        "gliner",
        "gliner_relex",
        "mlx_qwen3",
        "ollama_qwen3",
        "spacy",
        "transformer_ner",
        "transformers_qwen3",
    }
    assert config.extractors["mlx_qwen3"].parameters == {
        "required_profiles": ["darwin-arm64"]
    }
    assert config.llm.model == "qwen3:1.7b-q4_K_M"
    assert config.llm.structured_output is True
    assert config.llm.evidence_required is True


def test_complete_profile_does_not_advertise_research_or_stub_toggles() -> None:
    config = DataToKGConfig.from_yaml(_COMPLETE_CONFIG)
    configured = {
        name
        for techniques in config.enabled_techniques().values()
        for name in techniques
    }

    assert {
        "rebel",
        "supervised_re",
        "human_review",
        "external_kg_linking",
        "graph_based",
        "node_embeddings",
    }.isdisjoint(configured)
    assert config.enrichment == {}
    assert set(config.resolution) == {"exact"}


def test_every_declared_capability_is_mandatory() -> None:
    config = DataToKGConfig.from_yaml(_COMPLETE_CONFIG)

    groups = (
        config.loaders,
        config.mappers,
        config.extractors,
        config.resolution,
        config.validation,
        config.graph_store,
        config.enrichment,
        config.evaluation,
    )
    assert all(item.enabled is True for group in groups for item in group.values())
    with pytest.raises(ValidationError):
        DataToKGConfig.model_validate(
            {"extractors": {"gliner": {"enabled": False}}}
        )


def test_complete_profile_executes_formation_and_use_quality_contracts() -> None:
    config = DataToKGConfig.from_yaml(_COMPLETE_CONFIG)

    assert set(config.evaluation) == {
        "construction_metrics",
        "intended_use_metrics",
        "knowledge_graph_quality",
        "schema_quality",
        "validation_failure_metrics",
    }


def test_construct_profile_claims_only_graphs_it_materializes() -> None:
    config = DataToKGConfig.from_yaml(_COMPLETE_CONFIG)
    ingestion = yaml.safe_load(
        Path("configs/production_ingestion.example.yaml").read_text(encoding="utf-8")
    )["production_ingestion"]

    assert set(config.graph_store) == {"property_graph", "rdf", "evidence_graph"}
    assert {"neo4j", "ladybug"} <= set(ingestion)


def test_evaluation_example_runs_all_candidates_with_observed_runtime() -> None:
    config = TechniqueEvaluationConfig.from_yaml(_EVALUATION_CONFIG)

    assert config.evaluation.force_all_techniques is True
    assert config.evaluation.deterministic_runtime is False
    assert config.hard_gates.require_gold_dataset is True


@pytest.mark.parametrize(
    "payload",
    [
        {"unstructured": {"require_all_platform_providers": False}},
        {"unstructured": {"models_lock_path": ""}},
        {"unstructured": {"models_lock_path": "models.json"}},
        {"unstructured": {"cache_root": " "}},
        {
            "structured": {
                "require_declarations": True,
                "tables": [],
            }
        },
    ],
)
def test_mandatory_contract_rejects_partial_profiles(payload: object) -> None:
    with pytest.raises(ValidationError):
        DataToKGConfig.model_validate(payload)
