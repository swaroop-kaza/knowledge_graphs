"""Ollama-based evidence-grounded graph extraction."""

from __future__ import annotations

from typing import Any

from kg_construction.domain.evidence import span_evidence
from kg_construction.domain.models import (
    ConstructionBatch,
    EntityCandidate,
    RelationCandidate,
    SourceRecord,
)
from kg_construction.infrastructure.llm.ollama import OllamaProvider
from kg_construction.infrastructure.llm.prompts import (
    GRAPH_EXTRACTION_PROMPT_VERSION,
    build_graph_extraction_prompt,
    validate_graph_payload,
)

JsonDict = dict[str, Any]


class OllamaGraphExtractor:
    """Extract graph candidates from text through Ollama structured JSON."""

    name = "ollama_graph_extractor"
    technique = "llm_ollama_graph_extraction"

    def __init__(
        self,
        provider: OllamaProvider,
        *,
        allowed_node_types: list[str] | None = None,
        allowed_relationship_types: list[str] | None = None,
        schema_version: str = "v1",
    ) -> None:
        self.provider = provider
        self.allowed_node_types = allowed_node_types or ["Person", "Organization", "System", "Concept", "Event"]
        self.allowed_relationship_types = allowed_relationship_types or ["RELATED_TO", "MENTIONS", "OWNS", "USES"]
        self.schema_version = schema_version

    def extract(self, source: SourceRecord) -> ConstructionBatch:
        text = source.content or "\n".join(repr(row) for row in source.records)
        prompt = build_graph_extraction_prompt(
            chunk_text=text,
            allowed_node_types=self.allowed_node_types,
            allowed_relationship_types=self.allowed_relationship_types,
        )
        payload = self.provider.complete_json(prompt, validator=validate_graph_payload)
        entities: list[EntityCandidate] = []
        id_hint_to_entity_id: dict[str, str] = {}

        for node in payload.get("nodes", []):
            evidence_text = str(node.get("evidence_text", ""))
            evidence = self._evidence_from_text(source, text, evidence_text)
            if evidence is None:
                continue
            entity_type = str(node.get("type", "Concept"))
            if self.allowed_node_types and entity_type not in self.allowed_node_types:
                continue
            entity = EntityCandidate.build(
                entity_type=entity_type,
                name=str(node.get("name", node.get("id_hint", "unknown"))),
                properties={
                    **self._dict_value(node.get("properties")),
                    "id_hint": node.get("id_hint"),
                    "prompt_version": GRAPH_EXTRACTION_PROMPT_VERSION,
                },
                evidence=[evidence],
                confidence=float(node.get("confidence", 0.7)),
                extractor_name=self.name,
                technique=self.technique,
                schema_version=self.schema_version,
            )
            entities.append(entity)
            for key in {str(node.get("id_hint", "")), entity.name}:
                if key:
                    id_hint_to_entity_id[key] = entity.candidate_id

        relations: list[RelationCandidate] = []
        for relationship in payload.get("relationships", []):
            evidence_text = str(relationship.get("evidence_text", ""))
            evidence = self._evidence_from_text(source, text, evidence_text)
            if evidence is None:
                continue
            relation_type = str(relationship.get("type", "RELATED_TO"))
            if self.allowed_relationship_types and relation_type not in self.allowed_relationship_types:
                continue
            source_id = id_hint_to_entity_id.get(str(relationship.get("source")), str(relationship.get("source")))
            target_id = id_hint_to_entity_id.get(str(relationship.get("target")), str(relationship.get("target")))
            relations.append(
                RelationCandidate.build(
                    source=source_id,
                    relation_type=relation_type,
                    target=target_id,
                    properties={
                        **self._dict_value(relationship.get("properties")),
                        "prompt_version": GRAPH_EXTRACTION_PROMPT_VERSION,
                    },
                    evidence=[evidence],
                    confidence=float(relationship.get("confidence", 0.7)),
                    extractor_name=self.name,
                    technique=self.technique,
                    schema_version=self.schema_version,
                )
            )

        return ConstructionBatch(
            source_records=[source],
            entities=entities,
            relations=relations,
            metadata={
                "extractor": self.name,
                "llm_provider": self.provider.config.provider,
                "llm_model": self.provider.config.model,
                "prompt_version": GRAPH_EXTRACTION_PROMPT_VERSION,
            },
        )

    def _evidence_from_text(self, source: SourceRecord, text: str, evidence_text: str):
        if not evidence_text:
            return None
        start = text.find(evidence_text)
        if start < 0 and self.provider.config.evidence_required:
            return None
        if start < 0:
            start = 0
            end = 0
        else:
            end = start + len(evidence_text)
        return span_evidence(source, start=start, end=end, text=evidence_text)

    @staticmethod
    def _dict_value(value: object) -> JsonDict:
        return value if isinstance(value, dict) else {}


__all__ = ["OllamaGraphExtractor"]
