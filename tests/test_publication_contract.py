from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

from kg_construction.application.config import DataToKGConfig
from kg_construction.application.public_api import construct_kg, load_sources
from kg_construction.application.publication_contract import (
    GRAPH_BUILDER_CONTRACT_VERSION,
    PublicationContractRegistry,
    UntrustedPublicationContractError,
    publication_config_sha256,
    publication_source_catalog_sha256,
    source_snapshot_sha256,
)
from kg_construction.application.source_catalog import (
    SourceCatalog,
    TableDeclaration,
)
from kg_construction.domain.models import SourceRecord
from kg_construction.infrastructure.evaluation import VersionedGraphEvaluationSuite

_TEXT_SUITE = Path("configs/graph_evaluation_suite.sample_text.yaml")
_TEXT_SOURCE = Path("examples/data_to_kg/sample.txt")


def _write_suite(
    tmp_path: Path,
    suite: VersionedGraphEvaluationSuite,
    *,
    name: str = "suite.yaml",
) -> Path:
    path = tmp_path / name
    path.write_text(
        yaml.safe_dump(
            suite.model_dump(mode="json"),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_package_registry_authorizes_only_the_exact_reviewed_text_tuple() -> None:
    registry = PublicationContractRegistry.from_package_resource()
    suite = VersionedGraphEvaluationSuite.from_path(_TEXT_SUITE)
    sources = load_sources(_TEXT_SOURCE, source_type="text")
    config = DataToKGConfig()

    binding = registry.authorize(
        suite=suite,
        suite_path=_TEXT_SUITE,
        sources=sources,
        config=config,
    )

    assert binding.release_eligible
    assert not binding.research_only
    assert binding.contract_id == "sample-text-release-v2"
    assert binding.evaluation_fixture_sha256 == suite.fixture_sha256
    assert binding.source_snapshot_sha256 == (
        source_snapshot_sha256(sources[0]),
    )
    assert binding.runtime_source_snapshot_sha256 == (
        "d55b9889c2d0ba93c011287453391453d690ffabbc0c32a3e42521ab8d5aaef5",
    )
    assert binding.config_sha256 == publication_config_sha256(config)
    assert binding.source_catalog_sha256 == publication_source_catalog_sha256(
        config,
        source_catalog=None,
    )
    assert binding.quality_thresholds_version == "kg-quality-v3"
    assert len(binding.quality_thresholds_sha256) == 64
    assert (
        binding.graph_builder_contract_version
        == GRAPH_BUILDER_CONTRACT_VERSION
    )


def test_self_sealed_weakened_suite_is_explicitly_research_only(
    tmp_path: Path,
) -> None:
    registry = PublicationContractRegistry.from_package_resource()
    reviewed = VersionedGraphEvaluationSuite.from_path(_TEXT_SUITE)
    payload = reviewed.model_dump(mode="json", exclude={"fixture_sha256"})
    payload["graphrag"]["min_recall_at_k"] = 0.0
    weakened = VersionedGraphEvaluationSuite.seal(payload)
    path = _write_suite(tmp_path, weakened)

    assessment = registry.assess(
        suite=weakened,
        suite_path=path,
        sources=load_sources(_TEXT_SOURCE, source_type="text"),
        config=DataToKGConfig(),
    )

    assert assessment.research_only
    assert not assessment.release_eligible
    assert assessment.binding is None
    assert any("below floor" in reason for reason in assessment.reasons)
    with pytest.raises(
        UntrustedPublicationContractError,
        match="research-only",
    ):
        registry.authorize(
            suite=weakened,
            suite_path=path,
            sources=load_sources(_TEXT_SOURCE, source_type="text"),
            config=DataToKGConfig(),
        )


def test_custom_self_sealed_fixture_cannot_join_release_allowlist(
    tmp_path: Path,
) -> None:
    registry = PublicationContractRegistry.from_package_resource()
    reviewed = VersionedGraphEvaluationSuite.from_path(_TEXT_SUITE)
    payload = reviewed.model_dump(mode="json", exclude={"fixture_sha256"})
    payload["fixture_id"] = "custom-self-approved-suite"
    custom = VersionedGraphEvaluationSuite.seal(payload)
    path = _write_suite(tmp_path, custom)

    assessment = registry.assess(
        suite=custom,
        suite_path=path,
        sources=load_sources(_TEXT_SOURCE, source_type="text"),
        config=DataToKGConfig(),
    )

    assert assessment.research_only
    assert any("absent" in reason for reason in assessment.reasons)


def test_reviewed_suite_rejects_changed_source_and_configuration() -> None:
    registry = PublicationContractRegistry.from_package_resource()
    suite = VersionedGraphEvaluationSuite.from_path(_TEXT_SUITE)
    changed_source = SourceRecord.from_text(
        _TEXT_SOURCE.read_text(encoding="utf-8") + " Changed.",
        source_type="text",
        source_name=str(_TEXT_SOURCE),
    )
    changed_config = DataToKGConfig.model_validate(
        {"llm": {"max_retries": 4}}
    )

    source_assessment = registry.assess(
        suite=suite,
        suite_path=_TEXT_SUITE,
        sources=[changed_source],
        config=DataToKGConfig(),
    )
    config_assessment = registry.assess(
        suite=suite,
        suite_path=_TEXT_SUITE,
        sources=load_sources(_TEXT_SOURCE, source_type="text"),
        config=changed_config,
    )

    assert source_assessment.research_only
    assert any("source snapshot" in reason for reason in source_assessment.reasons)
    assert config_assessment.research_only
    assert any("configuration" in reason for reason in config_assessment.reasons)


def test_unreviewed_explicit_source_catalog_is_research_only() -> None:
    registry = PublicationContractRegistry.from_package_resource()
    suite = VersionedGraphEvaluationSuite.from_path(_TEXT_SUITE)
    catalog = SourceCatalog().declare(
        TableDeclaration(
            table_name="unreviewed.csv",
            record_type="Unreviewed",
            primary_key=["id"],
        )
    )

    assessment = registry.assess(
        suite=suite,
        suite_path=_TEXT_SUITE,
        sources=load_sources(_TEXT_SOURCE, source_type="text"),
        config=DataToKGConfig(),
        source_catalog=catalog,
    )

    assert assessment.research_only
    assert any("source-catalog" in reason for reason in assessment.reasons)


def test_registry_and_policy_are_content_addressed() -> None:
    registry = PublicationContractRegistry.from_package_resource()
    payload = registry.model_dump(mode="json")
    payload["contracts"][0]["fixture_id"] = "tampered"

    with pytest.raises(ValidationError, match="registry_sha256 mismatch"):
        PublicationContractRegistry.model_validate(payload)

    payload = registry.model_dump(mode="json")
    payload["policy"]["suite_minimums"]["graphrag.min_recall_at_k"] = 0.0
    with pytest.raises(ValidationError, match="policy_sha256 mismatch"):
        PublicationContractRegistry.model_validate(payload)


def test_public_api_refuses_untrusted_suite_before_runtime_or_graph_writes(
    tmp_path: Path,
) -> None:
    reviewed = VersionedGraphEvaluationSuite.from_path(_TEXT_SUITE)
    payload = reviewed.model_dump(mode="json", exclude={"fixture_sha256"})
    payload["fixture_id"] = "custom-public-attempt"
    custom = VersionedGraphEvaluationSuite.seal(payload)
    path = _write_suite(tmp_path, custom)

    with patch(
        "kg_construction.application.public_api.build_production_services"
    ) as services, pytest.raises(UntrustedPublicationContractError):
        construct_kg(
            _TEXT_SOURCE,
            source_type="text",
            evaluation_suite_path=path,
        )

    services.assert_not_called()
