"""Entity resolution implementations beyond exact matching."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.domain.models import ConstructionBatch, EntityCandidate
from kg_construction.infrastructure.loaders.stubs import AdapterNotConfigured
from kg_construction.infrastructure.resolution.common import (
    MergeDecision,
    apply_merge_decisions,
    compact_resolution_key,
    current_canonical_name,
    decisions_from_groups,
    resolution_key,
)


class NormalizedStringResolverConfig(BaseModel):
    """Config for deterministic normalized-string ER."""

    model_config = ConfigDict(extra="forbid")

    confidence: float = Field(default=0.98, ge=0.0, le=1.0)
    strip_suffix_tokens: list[str] = Field(default_factory=lambda: ["inc", "llc", "ltd", "corp", "corporation"])


class NormalizedStringResolver:
    """Resolve entities by punctuation, whitespace, case, and suffix normalization."""

    name = "normalized_string_resolver"
    technique = "normalized_string_matching"

    def __init__(self, config: NormalizedStringResolverConfig | None = None) -> None:
        self.config = config or NormalizedStringResolverConfig()

    def resolve(self, batch: ConstructionBatch) -> ConstructionBatch:
        decisions = decisions_from_groups(
            batch.entities,
            key_fn=lambda entity: self._key(current_canonical_name(entity)),
            resolver_name=self.name,
            confidence=self.config.confidence,
            reason="normalized_string_key",
        )
        return apply_merge_decisions(
            batch,
            decisions,
            resolver_name=self.name,
            resolver_technique=self.technique,
        )

    def _key(self, value: str) -> str:
        tokens = [
            token
            for token in resolution_key(value).split()
            if token not in {item.casefold() for item in self.config.strip_suffix_tokens}
        ]
        return " ".join(tokens)


class AliasTableResolverConfig(BaseModel):
    """Config for controlled alias-table entity resolution."""

    model_config = ConfigDict(extra="forbid")

    aliases: dict[str, list[str]] = Field(default_factory=dict)
    confidence: float = Field(default=0.99, ge=0.0, le=1.0)
    review_required: bool = False


class AliasTableResolver:
    """Resolve entities through a configured canonical-name to aliases table."""

    name = "alias_table_resolver"
    technique = "alias_table_matching"

    def __init__(
        self,
        aliases: dict[str, list[str]] | None = None,
        *,
        config: AliasTableResolverConfig | None = None,
    ) -> None:
        self.config = config or AliasTableResolverConfig(aliases=aliases or {})
        self.alias_to_canonical = self._alias_index(self.config.aliases)

    def resolve(self, batch: ConstructionBatch) -> ConstructionBatch:
        decisions: dict[str, MergeDecision] = {}
        for entity in batch.entities:
            prior_name = current_canonical_name(entity)
            raw_alias_key = compact_resolution_key(entity.name)
            prior_alias_key = compact_resolution_key(prior_name)
            canonical_name = self.alias_to_canonical.get(raw_alias_key)
            alias_matched = canonical_name is not None
            if not alias_matched:
                canonical_name = self.alias_to_canonical.get(prior_alias_key, prior_name)
            canonical_name = str(canonical_name)
            canonical_key = resolution_key(canonical_name)
            prior_confidence = float(entity.properties.get("resolution_confidence", self.config.confidence))
            prior_review_required = bool(entity.properties.get("review_required", self.config.review_required))
            decisions[entity.candidate_id] = MergeDecision(
                candidate_id=entity.candidate_id,
                canonical_key=canonical_key,
                canonical_name=canonical_name,
                confidence=self.config.confidence if alias_matched else prior_confidence,
                review_required=self.config.review_required if alias_matched else prior_review_required,
                reason="alias_table" if alias_matched else "alias_table_passthrough",
            )
        return apply_merge_decisions(
            batch,
            decisions,
            resolver_name=self.name,
            resolver_technique=self.technique,
        )

    @staticmethod
    def _alias_index(aliases: dict[str, list[str]]) -> dict[str, str]:
        index: dict[str, str] = {}
        for canonical_name, alias_values in aliases.items():
            index[compact_resolution_key(canonical_name)] = canonical_name
            for alias in alias_values:
                index[compact_resolution_key(alias)] = canonical_name
        return index


class RapidFuzzResolverConfig(BaseModel):
    """Config for RapidFuzz fuzzy entity resolution."""

    model_config = ConfigDict(extra="forbid")

    threshold: float = Field(default=90.0, ge=0.0, le=100.0)
    review_threshold: float = Field(default=96.0, ge=0.0, le=100.0)
    scorer: str = "WRatio"


class RapidFuzzResolver:
    """Resolve near-duplicate entity names with RapidFuzz string similarity."""

    name = "rapidfuzz_resolver"
    technique = "rapidfuzz_fuzzy_matching"

    def __init__(self, config: RapidFuzzResolverConfig | None = None) -> None:
        self.config = config or RapidFuzzResolverConfig()

    def resolve(self, batch: ConstructionBatch) -> ConstructionBatch:
        scorer = self._scorer()
        decisions: dict[str, MergeDecision] = {}
        entities_by_type: dict[str, list[EntityCandidate]] = {}
        for entity in batch.entities:
            entities_by_type.setdefault(entity.entity_type, []).append(entity)

        for entity_type, entities in entities_by_type.items():
            parent = {entity.candidate_id: entity.candidate_id for entity in entities}

            pair_scores: dict[tuple[str, str], float] = {}
            for left, right in combinations(entities, 2):
                score = float(
                    scorer(
                        resolution_key(current_canonical_name(left)),
                        resolution_key(current_canonical_name(right)),
                    )
                )
                if score >= self.config.threshold:
                    self._union(parent, left.candidate_id, right.candidate_id)
                    pair_key = tuple(sorted((left.candidate_id, right.candidate_id)))
                    pair_scores[(pair_key[0], pair_key[1])] = score

            clusters: dict[str, list[EntityCandidate]] = {}
            for entity in entities:
                clusters.setdefault(self._find(parent, entity.candidate_id), []).append(entity)

            for cluster in clusters.values():
                canonical = sorted(
                    cluster,
                    key=lambda item: (-len(current_canonical_name(item)), current_canonical_name(item).casefold()),
                )[0]
                canonical_name = current_canonical_name(canonical)
                canonical_key = resolution_key(canonical_name)
                for entity in cluster:
                    scores = [
                        score
                        for pair, score in pair_scores.items()
                        if entity.candidate_id in pair
                    ]
                    confidence = (min(scores) / 100.0) if scores else 1.0
                    decisions[entity.candidate_id] = MergeDecision(
                        candidate_id=entity.candidate_id,
                        canonical_key=f"{entity_type}:{canonical_key}",
                        canonical_name=canonical_name,
                        confidence=confidence,
                        review_required=confidence < (self.config.review_threshold / 100.0),
                        reason="rapidfuzz_similarity",
                    )

        return apply_merge_decisions(
            batch,
            decisions,
            resolver_name=self.name,
            resolver_technique=self.technique,
        )

    @staticmethod
    def _find(parent: dict[str, str], candidate_id: str) -> str:
        while parent[candidate_id] != candidate_id:
            parent[candidate_id] = parent[parent[candidate_id]]
            candidate_id = parent[candidate_id]
        return candidate_id

    @classmethod
    def _union(cls, parent: dict[str, str], left: str, right: str) -> None:
        left_root = cls._find(parent, left)
        right_root = cls._find(parent, right)
        if left_root != right_root:
            parent[right_root] = left_root

    def _scorer(self):
        try:
            from rapidfuzz import fuzz
        except ImportError as exc:
            raise AdapterNotConfigured(
                "RapidFuzz is not installed. Install the dedup extra or rapidfuzz>=3.9."
            ) from exc
        scorers = {
            "WRatio": fuzz.WRatio,
            "QRatio": fuzz.QRatio,
            "ratio": fuzz.ratio,
            "token_set_ratio": fuzz.token_set_ratio,
            "token_sort_ratio": fuzz.token_sort_ratio,
        }
        return scorers.get(self.config.scorer, fuzz.WRatio)


class SplinkResolverConfig(BaseModel):
    """Configuration placeholder for Splink probabilistic linkage."""

    model_config = ConfigDict(extra="forbid")

    settings: dict[str, Any] = Field(default_factory=dict)
    threshold_match_probability: float = Field(default=0.90, ge=0.0, le=1.0)


class SplinkProbabilisticResolver:
    """Adapter boundary for Splink probabilistic entity linkage."""

    name = "splink_probabilistic_resolver"
    technique = "splink_probabilistic_linkage"

    def __init__(self, config: SplinkResolverConfig | None = None) -> None:
        self.config = config or SplinkResolverConfig()

    def resolve(self, batch: ConstructionBatch) -> ConstructionBatch:
        try:
            import splink  # noqa: F401
        except ImportError as exc:
            raise AdapterNotConfigured(
                "Splink is not installed. Install the dedup extra and provide Splink settings."
            ) from exc
        raise AdapterNotConfigured(
            "Splink is available, but this adapter requires project-specific blocking rules, "
            "comparison columns, and a trained/estimated linker before resolving KG candidates."
        )


class DedupeTrainingBackend(Protocol):
    """Minimal protocol for an injected Dedupe backend."""

    def resolve(self, records: list[dict[str, object]]) -> list[dict[str, object]]:
        """Return rows containing candidate_id, canonical_key, confidence, and review_required."""


class DedupeResolverConfig(BaseModel):
    """Configuration for Dedupe active-learning entity resolution."""

    model_config = ConfigDict(extra="forbid")

    fields: list[str] = Field(default_factory=lambda: ["name", "entity_type"])
    confidence_threshold: float = Field(default=0.9, ge=0.0, le=1.0)


class DedupeActiveLearningResolver:
    """Adapter boundary for Dedupe active-learning entity resolution."""

    name = "dedupe_active_learning_resolver"
    technique = "dedupe_active_learning_entity_resolution"

    def __init__(
        self,
        config: DedupeResolverConfig | None = None,
        *,
        backend: DedupeTrainingBackend | None = None,
    ) -> None:
        self.config = config or DedupeResolverConfig()
        self.backend = backend

    def resolve(self, batch: ConstructionBatch) -> ConstructionBatch:
        if self.backend is None:
            try:
                import dedupe  # noqa: F401
            except ImportError as exc:
                raise AdapterNotConfigured(
                    "Dedupe is not installed. Install the dedup extra and configure active-learning training."
                ) from exc
            raise AdapterNotConfigured(
                "Dedupe is available, but active-learning ER needs labeled training pairs or "
                "an injected backend before automatic resolution."
            )

        rows = [
            {
                "candidate_id": entity.candidate_id,
                "name": entity.name,
                "entity_type": entity.entity_type,
                **entity.properties,
            }
            for entity in batch.entities
        ]
        backend_decisions = self.backend.resolve(rows)
        entity_by_id = {entity.candidate_id: entity for entity in batch.entities}
        decisions: dict[str, MergeDecision] = {}
        for item in backend_decisions:
            candidate_id = str(item.get("candidate_id", ""))
            entity = entity_by_id.get(candidate_id)
            if entity is None:
                continue
            raw_confidence = item.get("confidence", self.config.confidence_threshold)
            confidence = (
                float(raw_confidence)
                if isinstance(raw_confidence, int | float | str)
                else self.config.confidence_threshold
            )
            canonical_key = str(item.get("canonical_key", resolution_key(entity.name)))
            decisions[candidate_id] = MergeDecision(
                candidate_id=candidate_id,
                canonical_key=f"{entity.entity_type}:{canonical_key}",
                canonical_name=str(item.get("canonical_name", entity.name)),
                confidence=confidence,
                review_required=bool(item.get("review_required", confidence < self.config.confidence_threshold)),
                reason="dedupe_active_learning_backend",
            )
        return apply_merge_decisions(
            batch,
            decisions,
            resolver_name=self.name,
            resolver_technique=self.technique,
        )


__all__ = [
    "AliasTableResolver",
    "AliasTableResolverConfig",
    "DedupeActiveLearningResolver",
    "DedupeResolverConfig",
    "DedupeTrainingBackend",
    "NormalizedStringResolver",
    "NormalizedStringResolverConfig",
    "RapidFuzzResolver",
    "RapidFuzzResolverConfig",
    "SplinkProbabilisticResolver",
    "SplinkResolverConfig",
]
