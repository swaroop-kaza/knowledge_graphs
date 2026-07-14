"""External entity linking and semantic entity resolution adapters."""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from itertools import combinations
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.domain.models import ConstructionBatch, EntityCandidate
from kg_construction.infrastructure.loaders.stubs import AdapterNotConfigured
from kg_construction.infrastructure.resolution.common import (
    MergeDecision,
    apply_merge_decisions,
    current_canonical_name,
    resolution_key,
)

JsonDict = dict[str, Any]


class JsonTransport(Protocol):
    """Minimal injectable JSON transport for external linking services."""

    def __call__(
        self,
        url: str,
        *,
        params: JsonDict | None = None,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
        timeout_seconds: float = 30.0,
    ) -> JsonDict:
        """Return decoded JSON for an HTTP-like request."""


class EmbeddingModelProtocol(Protocol):
    """Minimal embedding model protocol."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text."""


class CrossEncoderModelProtocol(Protocol):
    """Minimal cross-encoder pair scoring protocol."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return one score per text pair."""


class WikidataLinkerConfig(BaseModel):
    """Configuration for Wikidata wbsearchentities linking."""

    model_config = ConfigDict(extra="forbid")

    endpoint_url: str = "https://www.wikidata.org/w/api.php"
    language: str = "en"
    limit: int = Field(default=5, ge=1, le=50)
    min_confidence: float = Field(default=0.72, ge=0.0, le=1.0)
    review_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    timeout_seconds: float = Field(default=30.0, gt=0.0)


class DBpediaLinkerConfig(BaseModel):
    """Configuration for DBpedia Lookup linking."""

    model_config = ConfigDict(extra="forbid")

    endpoint_url: str = "https://lookup.dbpedia.org/api/search"
    limit: int = Field(default=5, ge=1, le=50)
    min_confidence: float = Field(default=0.70, ge=0.0, le=1.0)
    review_threshold: float = Field(default=0.88, ge=0.0, le=1.0)
    timeout_seconds: float = Field(default=30.0, gt=0.0)


class OpenRefineReconcilerConfig(BaseModel):
    """Configuration for OpenRefine/W3C reconciliation services."""

    model_config = ConfigDict(extra="forbid")

    endpoint_url: str = "http://localhost:3333/reconcile"
    limit: int = Field(default=3, ge=1, le=50)
    type_map: dict[str, str] = Field(default_factory=dict)
    min_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    review_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    merge_on_match: bool = True


class EmbeddingResolverConfig(BaseModel):
    """Configuration for embedding similarity entity resolution."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    threshold: float = Field(default=0.86, ge=0.0, le=1.0)
    review_threshold: float = Field(default=0.94, ge=0.0, le=1.0)
    block_by_type: bool = True
    schema_version: str = "v1"


class CrossEncoderResolverConfig(BaseModel):
    """Configuration for cross-encoder pairwise entity resolution."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = "cross-encoder/stsb-roberta-base"
    threshold: float = Field(default=0.82, ge=0.0, le=1.0)
    review_threshold: float = Field(default=0.93, ge=0.0, le=1.0)
    block_by_type: bool = True
    max_pairs: int = Field(default=5000, ge=1)
    schema_version: str = "v1"


class WikidataLinker:
    """Attach Wikidata candidate links to entity candidates."""

    name = "wikidata_linker"
    technique = "wikidata_external_kb_linking"

    def __init__(
        self,
        config: WikidataLinkerConfig | None = None,
        *,
        transport: JsonTransport | None = None,
    ) -> None:
        self.config = config or WikidataLinkerConfig()
        self.transport = transport or http_json_transport

    def resolve(self, batch: ConstructionBatch) -> ConstructionBatch:
        linked_entities: list[EntityCandidate] = []
        audit: list[JsonDict] = []
        for entity in batch.entities:
            results = self._search(current_canonical_name(entity))
            candidates = [self._candidate(entity, item) for item in results]
            candidates = [item for item in candidates if item["confidence"] >= self.config.min_confidence]
            linked_entities.append(_attach_external_link(entity, "wikidata", candidates, self.name, self.technique))
            if candidates:
                audit.append(_external_link_audit(entity, candidates[0], "wikidata", self.name, self.technique))
        return _batch_with_external_audit(batch, linked_entities, audit, self.name)

    def _search(self, text: str) -> list[JsonDict]:
        payload = self.transport(
            self.config.endpoint_url,
            params={
                "action": "wbsearchentities",
                "format": "json",
                "language": self.config.language,
                "uselang": self.config.language,
                "type": "item",
                "limit": self.config.limit,
                "search": text,
            },
            timeout_seconds=self.config.timeout_seconds,
        )
        results = payload.get("search", [])
        return [dict(item) for item in results if isinstance(item, dict)]

    def _candidate(self, entity: EntityCandidate, item: JsonDict) -> JsonDict:
        label = str(item.get("label") or item.get("title") or "")
        identifier = str(item.get("id") or item.get("title") or "")
        url = str(item.get("concepturi") or item.get("url") or f"https://www.wikidata.org/wiki/{identifier}")
        confidence = _label_confidence(entity.name, label)
        return {
            "id": identifier,
            "label": label,
            "description": str(item.get("description", "")),
            "url": url,
            "confidence": confidence,
            "review_required": confidence < self.config.review_threshold,
            "raw": item,
        }


class DBpediaLinker:
    """Attach DBpedia Lookup candidate links to entity candidates."""

    name = "dbpedia_linker"
    technique = "dbpedia_external_kb_linking"

    def __init__(
        self,
        config: DBpediaLinkerConfig | None = None,
        *,
        transport: JsonTransport | None = None,
    ) -> None:
        self.config = config or DBpediaLinkerConfig()
        self.transport = transport or http_json_transport

    def resolve(self, batch: ConstructionBatch) -> ConstructionBatch:
        linked_entities: list[EntityCandidate] = []
        audit: list[JsonDict] = []
        for entity in batch.entities:
            results = self._search(current_canonical_name(entity))
            candidates = [self._candidate(entity, item) for item in results]
            candidates = [item for item in candidates if item["confidence"] >= self.config.min_confidence]
            linked_entities.append(_attach_external_link(entity, "dbpedia", candidates, self.name, self.technique))
            if candidates:
                audit.append(_external_link_audit(entity, candidates[0], "dbpedia", self.name, self.technique))
        return _batch_with_external_audit(batch, linked_entities, audit, self.name)

    def _search(self, text: str) -> list[JsonDict]:
        payload = self.transport(
            self.config.endpoint_url,
            params={"query": text, "maxResults": self.config.limit},
            headers={"Accept": "application/json"},
            timeout_seconds=self.config.timeout_seconds,
        )
        if isinstance(payload.get("docs"), list):
            return [dict(item) for item in payload["docs"] if isinstance(item, dict)]
        if isinstance(payload.get("results"), list):
            return [dict(item) for item in payload["results"] if isinstance(item, dict)]
        return []

    def _candidate(self, entity: EntityCandidate, item: JsonDict) -> JsonDict:
        label = _first_value(item.get("label") or item.get("label.value") or item.get("name"))
        resource = _first_value(item.get("resource") or item.get("uri") or item.get("id"))
        comment = _first_value(item.get("comment") or item.get("description"))
        confidence = _label_confidence(entity.name, label)
        return {
            "id": resource,
            "label": label,
            "description": comment,
            "url": resource,
            "confidence": confidence,
            "review_required": confidence < self.config.review_threshold,
            "raw": item,
        }


class OpenRefineReconciler:
    """Reconcile entities through an OpenRefine/W3C reconciliation service."""

    name = "openrefine_reconciler"
    technique = "openrefine_style_reconciliation"

    def __init__(
        self,
        config: OpenRefineReconcilerConfig | None = None,
        *,
        transport: JsonTransport | None = None,
    ) -> None:
        self.config = config or OpenRefineReconcilerConfig()
        self.transport = transport or http_json_transport

    def resolve(self, batch: ConstructionBatch) -> ConstructionBatch:
        query_index: dict[str, EntityCandidate] = {}
        queries: JsonDict = {}
        for index, entity in enumerate(batch.entities):
            query_id = f"q{index}"
            query_index[query_id] = entity
            query: JsonDict = {
                "query": current_canonical_name(entity),
                "limit": self.config.limit,
            }
            if type_id := self.config.type_map.get(entity.entity_type):
                query["type"] = type_id
            queries[query_id] = query

        payload = self.transport(
            self.config.endpoint_url,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
            body=urllib.parse.urlencode({"queries": json.dumps(queries)}).encode("utf-8"),
            timeout_seconds=self.config.timeout_seconds,
        )

        prelinked_entities: list[EntityCandidate] = []
        decisions: dict[str, MergeDecision] = {}
        audit: list[JsonDict] = []
        for query_id, entity in query_index.items():
            result_payload = payload.get(query_id, {})
            results = result_payload.get("result", []) if isinstance(result_payload, dict) else []
            candidates = [
                self._candidate(entity, item)
                for item in results
                if isinstance(item, dict)
            ]
            candidates = [item for item in candidates if item["confidence"] >= self.config.min_confidence]
            prelinked_entities.append(
                _attach_external_link(entity, "openrefine", candidates, self.name, self.technique)
            )
            if not candidates:
                continue
            best = candidates[0]
            audit.append(_external_link_audit(entity, best, "openrefine", self.name, self.technique))
            if self.config.merge_on_match:
                decisions[entity.candidate_id] = MergeDecision(
                    candidate_id=entity.candidate_id,
                    canonical_key=f"openrefine:{best['id']}",
                    canonical_name=str(best["label"] or entity.name),
                    confidence=float(best["confidence"]),
                    review_required=bool(best["review_required"]),
                    reason="openrefine_reconciliation_match",
                )

        prelinked_batch = batch.model_copy(
            update={
                "entities": prelinked_entities,
                "metadata": {
                    **batch.metadata,
                    "external_linking_audit": [*list(batch.metadata.get("external_linking_audit", [])), *audit],
                    "resolver": self.name,
                },
            }
        )
        if decisions:
            return apply_merge_decisions(
                prelinked_batch,
                decisions,
                resolver_name=self.name,
                resolver_technique=self.technique,
            )
        return prelinked_batch

    def _candidate(self, entity: EntityCandidate, item: JsonDict) -> JsonDict:
        label = str(item.get("name") or item.get("label") or "")
        identifier = str(item.get("id") or item.get("identifier") or "")
        raw_score = item.get("score", 0.0)
        score = _score_to_unit(raw_score)
        confidence = max(score, _label_confidence(entity.name, label) if label else 0.0)
        return {
            "id": identifier,
            "label": label,
            "description": str(item.get("description", "")),
            "type": item.get("type", []),
            "url": str(item.get("url", "")),
            "confidence": confidence,
            "review_required": confidence < self.config.review_threshold or not bool(item.get("match", False)),
            "match": bool(item.get("match", False)),
            "raw": item,
        }


class EmbeddingResolver:
    """Resolve entities using vector embedding cosine similarity."""

    name = "embedding_resolver"
    technique = "embedding_similarity_matching"

    def __init__(
        self,
        config: EmbeddingResolverConfig | None = None,
        *,
        model: EmbeddingModelProtocol | None = None,
    ) -> None:
        self.config = config or EmbeddingResolverConfig()
        self.model = model

    def resolve(self, batch: ConstructionBatch) -> ConstructionBatch:
        model = self._model()
        texts = [_entity_resolution_text(entity) for entity in batch.entities]
        vectors = model.encode(texts)
        pair_scores = _embedding_pair_scores(batch.entities, vectors, block_by_type=self.config.block_by_type)
        decisions = _decisions_from_pair_scores(
            batch.entities,
            pair_scores,
            threshold=self.config.threshold,
            review_threshold=self.config.review_threshold,
            reason="embedding_cosine_similarity",
        )
        return apply_merge_decisions(
            batch,
            decisions,
            resolver_name=self.name,
            resolver_technique=self.technique,
        )

    def _model(self) -> EmbeddingModelProtocol:
        if self.model is not None:
            return self.model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise AdapterNotConfigured(
                "Embedding resolution requires sentence-transformers or an injected model."
            ) from exc
        self.model = SentenceTransformer(self.config.model_name)
        return self.model


class CrossEncoderResolver:
    """Resolve entities using pairwise cross-encoder scores."""

    name = "cross_encoder_resolver"
    technique = "cross_encoder_pair_scoring"

    def __init__(
        self,
        config: CrossEncoderResolverConfig | None = None,
        *,
        model: CrossEncoderModelProtocol | None = None,
    ) -> None:
        self.config = config or CrossEncoderResolverConfig()
        self.model = model

    def resolve(self, batch: ConstructionBatch) -> ConstructionBatch:
        model = self._model()
        pairs: list[tuple[EntityCandidate, EntityCandidate]] = []
        text_pairs: list[tuple[str, str]] = []
        for left, right in _candidate_pairs(batch.entities, block_by_type=self.config.block_by_type):
            pairs.append((left, right))
            text_pairs.append((_entity_resolution_text(left), _entity_resolution_text(right)))
            if len(pairs) >= self.config.max_pairs:
                break
        scores = model.predict(text_pairs)
        pair_scores = {
            (left.candidate_id, right.candidate_id): _score_to_unit(score)
            for (left, right), score in zip(pairs, scores, strict=False)
        }
        decisions = _decisions_from_pair_scores(
            batch.entities,
            pair_scores,
            threshold=self.config.threshold,
            review_threshold=self.config.review_threshold,
            reason="cross_encoder_pair_score",
        )
        return apply_merge_decisions(
            batch,
            decisions,
            resolver_name=self.name,
            resolver_technique=self.technique,
        )

    def _model(self) -> CrossEncoderModelProtocol:
        if self.model is not None:
            return self.model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise AdapterNotConfigured(
                "Cross-encoder resolution requires sentence-transformers or an injected model."
            ) from exc
        self.model = CrossEncoder(self.config.model_name)
        return self.model


def http_json_transport(
    url: str,
    *,
    params: JsonDict | None = None,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    timeout_seconds: float = 30.0,
) -> JsonDict:
    """Small urllib JSON transport used by external linkers."""

    target = url
    if params:
        separator = "&" if "?" in target else "?"
        target = f"{target}{separator}{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        target,
        data=body,
        method=method,
        headers=headers or {"Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return json.loads(response.read().decode(charset))


def _attach_external_link(
    entity: EntityCandidate,
    provider: str,
    candidates: list[JsonDict],
    resolver_name: str,
    resolver_technique: str,
) -> EntityCandidate:
    if not candidates:
        return entity
    best = candidates[0]
    existing_links = list(entity.properties.get("external_links", []))
    properties = {
        **entity.properties,
        "external_links": [
            *existing_links,
            {
                "provider": provider,
                "id": best["id"],
                "label": best["label"],
                "url": best.get("url", ""),
                "confidence": round(float(best["confidence"]), 6),
                "review_required": bool(best["review_required"]),
                "candidates": candidates,
            },
        ],
        f"{provider}_id": best["id"],
        f"{provider}_label": best["label"],
        f"{provider}_url": best.get("url", ""),
        f"{provider}_confidence": round(float(best["confidence"]), 6),
        f"{provider}_review_required": bool(best["review_required"]),
        "resolver": resolver_name,
        "resolution_method": resolver_technique,
    }
    return entity.model_copy(update={"properties": properties})


def _batch_with_external_audit(
    batch: ConstructionBatch,
    entities: list[EntityCandidate],
    audit: list[JsonDict],
    resolver_name: str,
) -> ConstructionBatch:
    return batch.model_copy(
        update={
            "entities": entities,
            "metadata": {
                **batch.metadata,
                "external_linking_audit": [*list(batch.metadata.get("external_linking_audit", [])), *audit],
                "resolver": resolver_name,
            },
        }
    )


def _external_link_audit(
    entity: EntityCandidate,
    best: JsonDict,
    provider: str,
    resolver_name: str,
    technique: str,
) -> JsonDict:
    return {
        "candidate_id": entity.candidate_id,
        "entity_name": entity.name,
        "provider": provider,
        "external_id": best.get("id", ""),
        "external_label": best.get("label", ""),
        "external_url": best.get("url", ""),
        "confidence": round(float(best.get("confidence", 0.0)), 6),
        "review_required": bool(best.get("review_required", True)),
        "resolver": resolver_name,
        "method": technique,
    }


def _first_value(value: object) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    if isinstance(value, dict):
        return str(value.get("value", ""))
    return str(value or "")


def _label_confidence(left: str, right: str) -> float:
    left_key = resolution_key(left)
    right_key = resolution_key(right)
    if not left_key or not right_key:
        return 0.0
    if left_key == right_key:
        return 1.0
    try:
        from rapidfuzz import fuzz
    except ImportError:
        left_tokens = set(left_key.split())
        right_tokens = set(right_key.split())
        overlap = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
        contains = 0.9 if left_key in right_key or right_key in left_key else 0.0
        return max(overlap, contains)
    return float(fuzz.WRatio(left_key, right_key)) / 100.0


def _score_to_unit(value: object) -> float:
    if isinstance(value, int | float | str):
        try:
            score = float(value)
        except ValueError:
            return 0.0
        if score > 1.0:
            return min(score / 100.0, 1.0)
        return max(score, 0.0)
    return 0.0


def _entity_resolution_text(entity: EntityCandidate) -> str:
    description = entity.properties.get("description") or entity.properties.get("label") or ""
    return f"{entity.entity_type}: {current_canonical_name(entity)} {description}".strip()


def _candidate_pairs(
    entities: list[EntityCandidate],
    *,
    block_by_type: bool,
) -> list[tuple[EntityCandidate, EntityCandidate]]:
    if not block_by_type:
        return list(combinations(entities, 2))
    pairs: list[tuple[EntityCandidate, EntityCandidate]] = []
    entities_by_type: dict[str, list[EntityCandidate]] = {}
    for entity in entities:
        entities_by_type.setdefault(entity.entity_type, []).append(entity)
    for typed_entities in entities_by_type.values():
        pairs.extend(combinations(typed_entities, 2))
    return pairs


def _embedding_pair_scores(
    entities: list[EntityCandidate],
    vectors: list[list[float]],
    *,
    block_by_type: bool,
) -> dict[tuple[str, str], float]:
    vector_by_id = {
        entity.candidate_id: vector
        for entity, vector in zip(entities, vectors, strict=False)
    }
    scores: dict[tuple[str, str], float] = {}
    for left, right in _candidate_pairs(entities, block_by_type=block_by_type):
        scores[(left.candidate_id, right.candidate_id)] = _cosine(
            vector_by_id.get(left.candidate_id, []),
            vector_by_id.get(right.candidate_id, []),
        )
    return scores


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _decisions_from_pair_scores(
    entities: list[EntityCandidate],
    pair_scores: dict[tuple[str, str], float],
    *,
    threshold: float,
    review_threshold: float,
    reason: str,
) -> dict[str, MergeDecision]:
    parent = {entity.candidate_id: entity.candidate_id for entity in entities}
    score_lookup: dict[tuple[str, str], float] = {}
    for (left, right), score in pair_scores.items():
        if score < threshold:
            continue
        _union(parent, left, right)
        left_key, right_key = sorted((left, right))
        score_lookup[(left_key, right_key)] = score

    clusters: dict[str, list[EntityCandidate]] = {}
    for entity in entities:
        clusters.setdefault(_find(parent, entity.candidate_id), []).append(entity)

    decisions: dict[str, MergeDecision] = {}
    for cluster in clusters.values():
        if len(cluster) <= 1:
            continue
        canonical = sorted(
            cluster,
            key=lambda item: (-len(current_canonical_name(item)), current_canonical_name(item).casefold()),
        )[0]
        canonical_name = current_canonical_name(canonical)
        canonical_key = resolution_key(canonical_name)
        for entity in cluster:
            scores = [
                score
                for pair, score in score_lookup.items()
                if entity.candidate_id in pair
            ]
            confidence = min(scores) if scores else 1.0
            decisions[entity.candidate_id] = MergeDecision(
                candidate_id=entity.candidate_id,
                canonical_key=f"{entity.entity_type}:{canonical_key}",
                canonical_name=canonical_name,
                confidence=confidence,
                review_required=confidence < review_threshold,
                reason=reason,
            )
    return decisions


def _find(parent: dict[str, str], candidate_id: str) -> str:
    while parent[candidate_id] != candidate_id:
        parent[candidate_id] = parent[parent[candidate_id]]
        candidate_id = parent[candidate_id]
    return candidate_id


def _union(parent: dict[str, str], left: str, right: str) -> None:
    left_root = _find(parent, left)
    right_root = _find(parent, right)
    if left_root != right_root:
        parent[right_root] = left_root


__all__ = [
    "CrossEncoderModelProtocol",
    "CrossEncoderResolver",
    "CrossEncoderResolverConfig",
    "DBpediaLinker",
    "DBpediaLinkerConfig",
    "EmbeddingModelProtocol",
    "EmbeddingResolver",
    "EmbeddingResolverConfig",
    "JsonTransport",
    "OpenRefineReconciler",
    "OpenRefineReconcilerConfig",
    "WikidataLinker",
    "WikidataLinkerConfig",
    "http_json_transport",
]
