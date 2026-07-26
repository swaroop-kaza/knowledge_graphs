"""Mandatory local canonical-identity ensemble with an explicit NIL policy."""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.domain.claim import CanonicalEntity, Mention, ResolutionStatus
from kg_construction.domain.models import normalize_text, stable_id
from kg_construction.infrastructure.extractors.model_backed.contracts import (
    HydratedArtifact,
    MandatoryProviderError,
    OfflineArtifactError,
)
from kg_construction.infrastructure.resolution.linking import (
    CrossEncoderModelProtocol,
    EmbeddingModelProtocol,
)
from kg_construction.unstructured.pipeline import IdentityResolution


class NilIdentityPolicy(StrEnum):
    """What publication does when no governed external identity is accepted."""

    RETAIN_OBSERVED = "retain_observed"


class IdentityVote(BaseModel):
    """One provider's auditable assessment of a coreference cluster."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    external_system: str | None = None
    external_id: str | None = None


class ClusterIdentityProvider(Protocol):
    """Provider boundary used by the mandatory local identity ensemble."""

    provider_id: str

    def assess(
        self,
        canonical_name: str,
        entity_type: str,
        mentions: tuple[Mention, ...],
    ) -> IdentityVote: ...


class ExactClusterIdentityProvider:
    """Deterministic whole-cluster identity voter."""

    provider_id = "identity_exact_nfkc"

    def assess(
        self,
        canonical_name: str,
        entity_type: str,
        mentions: tuple[Mention, ...],
    ) -> IdentityVote:
        del entity_type
        normalized_anchor = normalize_text(canonical_name)
        if not normalized_anchor or not mentions:
            raise ValueError("exact identity assessment requires a non-empty cluster")
        compatible = all(
            normalize_text(mention.coreference_anchor or mention.surface)
            == normalized_anchor
            for mention in mentions
        )
        return IdentityVote(
            provider_id=self.provider_id,
            canonical_name=canonical_name,
            confidence=1.0 if compatible else 0.0,
        )


class LockedEmbeddingIdentityProvider:
    """Embedding agreement loaded only from the exact hydrated snapshot path."""

    provider_id = "identity_embedding"

    def __init__(
        self,
        artifact: HydratedArtifact,
        *,
        model: EmbeddingModelProtocol | None = None,
    ) -> None:
        if artifact.backend != "sentence-transformers":
            raise OfflineArtifactError(
                "identity embedding artifact must use sentence-transformers"
            )
        self.artifact = artifact
        self.model = model

    def assess(
        self,
        canonical_name: str,
        entity_type: str,
        mentions: tuple[Mention, ...],
    ) -> IdentityVote:
        del entity_type
        values = [canonical_name, *(mention.surface for mention in mentions)]
        vectors = self._model().encode(values)
        if len(vectors) != len(values):
            raise ValueError("embedding identity provider returned the wrong vector count")
        scores = [
            _cosine(vectors[0], vector)
            for vector in vectors[1:]
        ]
        return IdentityVote(
            provider_id=self.provider_id,
            canonical_name=canonical_name,
            confidence=min(scores, default=1.0),
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Expose the same locked model to the post-extraction resolver."""

        return self._model().encode(texts)

    def _model(self) -> EmbeddingModelProtocol:
        if self.model is not None:
            return self.model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - mandatory dependency
            raise OfflineArtifactError(
                "sentence-transformers is required for identity embeddings"
            ) from exc
        self.model = SentenceTransformer(
            str(self.artifact.local_path),
            device=self.artifact.device,
            local_files_only=True,
        )
        return self.model


class LockedCrossEncoderIdentityProvider:
    """Pair scorer loaded only from the exact hydrated cross-encoder snapshot."""

    provider_id = "identity_cross_encoder"

    def __init__(
        self,
        artifact: HydratedArtifact,
        *,
        model: CrossEncoderModelProtocol | None = None,
    ) -> None:
        if artifact.backend != "sentence-transformers":
            raise OfflineArtifactError(
                "identity cross-encoder artifact must use sentence-transformers"
            )
        self.artifact = artifact
        self.model = model

    def assess(
        self,
        canonical_name: str,
        entity_type: str,
        mentions: tuple[Mention, ...],
    ) -> IdentityVote:
        del entity_type
        pairs = [(canonical_name, mention.surface) for mention in mentions]
        scores = list(self._model().predict(pairs))
        if len(scores) != len(pairs):
            raise ValueError("cross-encoder identity provider returned the wrong score count")
        return IdentityVote(
            provider_id=self.provider_id,
            canonical_name=canonical_name,
            confidence=min((_unit_score(value) for value in scores), default=1.0),
        )

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Expose the same locked model to the post-extraction resolver."""

        return self._model().predict(pairs)

    def _model(self) -> CrossEncoderModelProtocol:
        if self.model is not None:
            return self.model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - mandatory dependency
            raise OfflineArtifactError(
                "sentence-transformers is required for cross-encoder identity resolution"
            ) from exc
        self.model = CrossEncoder(
            str(self.artifact.local_path),
            device=self.artifact.device,
            local_files_only=True,
        )
        return self.model


class MandatoryCanonicalIdentityEnsemble:
    """Run every identity provider and create governed local IDs for NIL clusters."""

    name = "mandatory_canonical_identity_ensemble"
    technique = "exact_embedding_cross_encoder_identity_with_explicit_nil"

    def __init__(
        self,
        providers: Sequence[ClusterIdentityProvider],
        *,
        schema_id: str,
        nil_policy: NilIdentityPolicy = NilIdentityPolicy.RETAIN_OBSERVED,
        minimum_consensus_confidence: float = 0.0,
    ) -> None:
        if not providers:
            raise ValueError("mandatory identity ensemble requires providers")
        if nil_policy != NilIdentityPolicy.RETAIN_OBSERVED:
            raise ValueError("the product identity ensemble requires retain_observed NIL policy")
        if not 0.0 <= minimum_consensus_confidence <= 1.0:
            raise ValueError("minimum identity consensus confidence must be in [0, 1]")
        self.providers = tuple(providers)
        self.provider_ids = tuple(provider.provider_id for provider in self.providers)
        if len(self.provider_ids) != len(set(self.provider_ids)):
            raise ValueError("mandatory identity provider IDs must be unique")
        self.schema_id = schema_id
        self.nil_policy = nil_policy
        self.minimum_consensus_confidence = minimum_consensus_confidence
        self.last_participation_ids: tuple[str, ...] = ()

    def resolve(self, mentions: tuple[Mention, ...]) -> IdentityResolution:
        grouped: dict[str, list[Mention]] = {}
        for mention in mentions:
            key = mention.coreference_cluster_id or mention.mention_id
            grouped.setdefault(key, []).append(mention)

        resolved_by_id: dict[str, Mention] = {}
        entities: list[CanonicalEntity] = []
        nil_clusters: list[str] = []
        decisions: list[dict[str, object]] = []
        participated: list[str] = []
        for cluster_id, members_list in sorted(grouped.items()):
            members = tuple(members_list)
            anchor = members[0].coreference_anchor or members[0].surface
            entity_type = members[0].entity_type
            votes: list[IdentityVote] = []
            failures: dict[str, BaseException] = {}
            for provider in self.providers:
                participated.append(provider.provider_id)
                try:
                    votes.append(provider.assess(anchor, entity_type, members))
                except Exception as exc:  # noqa: BLE001 - every provider runs before hard fail
                    failures[provider.provider_id] = exc
            if failures:
                self.last_participation_ids = tuple(dict.fromkeys(participated))
                raise MandatoryProviderError("canonical identity resolution", failures)

            confidence = sum(vote.confidence for vote in votes) / len(votes)
            if confidence < self.minimum_consensus_confidence:
                raise ValueError(
                    f"identity consensus {confidence:.3f} is below "
                    f"{self.minimum_consensus_confidence:.3f} for {anchor!r}"
                )
            external_votes = [
                vote
                for vote in votes
                if vote.external_system is not None and vote.external_id is not None
            ]
            chosen_external = max(
                external_votes,
                key=lambda vote: (vote.confidence, vote.provider_id),
                default=None,
            )
            if chosen_external is None:
                nil_clusters.append(cluster_id)
                decisions.append(
                    {
                        "cluster_id": cluster_id,
                        "canonical_entity_id": None,
                        "canonical_name": anchor,
                        "external_match": False,
                        "nil_policy": self.nil_policy.value,
                        "provider_ids": list(self.provider_ids),
                        "votes": [
                            {
                                "provider_id": vote.provider_id,
                                "confidence": vote.confidence,
                                "canonical_name": vote.canonical_name,
                                "external_system": vote.external_system,
                                "external_id": vote.external_id,
                            }
                            for vote in votes
                        ],
                    }
                )
                for member in members:
                    resolved_by_id[member.mention_id] = member.model_copy(
                        update={
                            "resolution_status": (
                                ResolutionStatus.COREFERENCE_ONLY
                                if member.coreference_cluster_id is not None
                                else ResolutionStatus.UNRESOLVED
                            ),
                            "resolution_confidence": confidence,
                        }
                    )
                continue
            else:
                entity_id = stable_id(
                    "canonical",
                    chosen_external.external_system,
                    chosen_external.external_id,
                )
                canonical_name = chosen_external.canonical_name

            entities.append(
                CanonicalEntity(
                    entity_id=entity_id,
                    canonical_name=canonical_name,
                    entity_type=entity_type,
                    mention_ids=tuple(member.mention_id for member in members),
                    aliases=tuple(
                        sorted({member.surface for member in members} - {canonical_name})
                    ),
                    external_system=chosen_external.external_system,
                    external_id=chosen_external.external_id,
                    confidence=confidence,
                )
            )
            decisions.append(
                {
                    "cluster_id": cluster_id,
                    "canonical_entity_id": entity_id,
                    "canonical_name": canonical_name,
                    "external_match": chosen_external is not None,
                    "nil_policy": self.nil_policy.value,
                    "provider_ids": list(self.provider_ids),
                    "votes": [
                        {
                            "provider_id": vote.provider_id,
                            "confidence": vote.confidence,
                            "canonical_name": vote.canonical_name,
                            "external_system": vote.external_system,
                            "external_id": vote.external_id,
                        }
                        for vote in votes
                    ],
                }
            )
            for member in members:
                resolved_by_id[member.mention_id] = member.model_copy(
                    update={
                        "canonical_entity_id": entity_id,
                        "resolution_status": ResolutionStatus.LINKED,
                        "resolution_confidence": confidence,
                    }
                )
        self.last_participation_ids = tuple(dict.fromkeys(participated))
        return IdentityResolution(
            mentions=tuple(resolved_by_id[item.mention_id] for item in mentions),
            canonical_entities=tuple(entities),
            provider_participation_ids=self.last_participation_ids,
            nil_policy=self.nil_policy.value,
            nil_cluster_ids=tuple(nil_clusters),
            decisions=tuple(decisions),
        )


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    # SentenceTransformers returns NumPy arrays in live execution.  Testing an
    # array directly for truth is ambiguous, so dimension checks must remain
    # sequence-generic.
    if len(left) != len(right) or len(left) == 0:
        raise ValueError("identity vectors must have the same non-zero dimension")
    numerator = sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(0.0, min(1.0, numerator / (left_norm * right_norm)))


def _unit_score(value: float) -> float:
    score = float(value)
    if 0.0 <= score <= 1.0:
        return score
    return 1.0 / (1.0 + math.exp(-score))


__all__ = [
    "ClusterIdentityProvider",
    "ExactClusterIdentityProvider",
    "IdentityVote",
    "LockedCrossEncoderIdentityProvider",
    "LockedEmbeddingIdentityProvider",
    "MandatoryCanonicalIdentityEnsemble",
    "NilIdentityPolicy",
]
