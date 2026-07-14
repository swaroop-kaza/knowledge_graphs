"""Convenience pipeline factory."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kg_construction.application.config import DataToKGConfig
from kg_construction.application.workflow import ConstructionWorkflow
from kg_construction.infrastructure.extractors.regex import RegexEntityExtractor
from kg_construction.infrastructure.mapping.direct_table_mapper import DirectTableMapper
from kg_construction.infrastructure.ontology.validation import EvidenceRequiredValidator
from kg_construction.infrastructure.resolution.entity_resolution import (
    AliasTableResolver,
    NormalizedStringResolver,
    RapidFuzzResolver,
    RapidFuzzResolverConfig,
)
from kg_construction.infrastructure.resolution.exact import ExactKeyEntityResolver


class DataToKGPipeline:
    """Build a default local pipeline from YAML configuration."""

    @classmethod
    def from_config(cls, config: DataToKGConfig) -> ConstructionWorkflow:
        mappers: list[object] = []
        extractors: list[object] = []
        resolvers: list[object] = []
        validators: list[object] = [EvidenceRequiredValidator(config.pipeline.validation_policy)]

        if _enabled(config.mappers, "direct_table", "direct_table_mapper"):
            mappers.append(DirectTableMapper())
        if _enabled(config.extractors, "regex", "regex_entity_extractor"):
            extractors.append(RegexEntityExtractor())
        if _enabled(config.resolution, "exact", "exact_key_entity_resolver"):
            resolvers.append(ExactKeyEntityResolver())
        if _enabled(config.resolution, "normalized", "normalized_string_resolver"):
            resolvers.append(NormalizedStringResolver())
        if _enabled(config.resolution, "alias", "alias_table_resolver"):
            aliases = _parameters(config.resolution, "alias", "alias_table_resolver").get("aliases", {})
            resolvers.append(AliasTableResolver(aliases=aliases if isinstance(aliases, dict) else {}))
        if _enabled(config.resolution, "rapidfuzz", "fuzzy", "rapidfuzz_resolver", "fuzzy_string_resolver"):
            params = _parameters(config.resolution, "rapidfuzz", "fuzzy", "rapidfuzz_resolver", "fuzzy_string_resolver")
            resolvers.append(RapidFuzzResolver(RapidFuzzResolverConfig.model_validate(params)))

        return ConstructionWorkflow(
            mappers=mappers,
            extractors=extractors,
            resolvers=resolvers,
            validators=validators,
        )


def _enabled(group: Mapping[str, Any], *names: str) -> bool:
    for name in names:
        item = group.get(name)
        if item is not None and bool(getattr(item, "enabled", False)):
            return True
    return False


def _parameters(group: Mapping[str, Any], *names: str) -> dict[str, Any]:
    for name in names:
        item = group.get(name)
        if item is not None:
            parameters = getattr(item, "parameters", {})
            return parameters if isinstance(parameters, dict) else {}
    return {}


__all__ = ["DataToKGPipeline"]
