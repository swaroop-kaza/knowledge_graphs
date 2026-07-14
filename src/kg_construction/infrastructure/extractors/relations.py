"""Evidence-backed relation extraction techniques."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field

from kg_construction.domain.evidence import span_evidence
from kg_construction.domain.models import (
    ConstructionBatch,
    EntityCandidate,
    RelationCandidate,
    SourceRecord,
    normalize_text,
)
from kg_construction.infrastructure.extractors.gliner import (
    GlinerConfig,
    GlinerEntityExtractor,
    GlinerModelProtocol,
)
from kg_construction.infrastructure.llm.ollama import OllamaProvider
from kg_construction.infrastructure.llm.prompts import (
    DEFAULT_RELATION_TYPING_TEMPLATE,
    RELATION_TYPING_PROMPT_VERSION,
    build_relation_typing_prompt,
    validate_relation_typing_payload,
)
from kg_construction.infrastructure.loaders.stubs import AdapterNotConfigured

JsonDict = dict[str, Any]

_TOKEN = r"[A-Za-z0-9][A-Za-z0-9_.:/#@+-]*"
_ARGUMENT = rf"{_TOKEN}(?:\s+{_TOKEN}){{0,7}}?"
_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]|$)")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_IDENTIFIER_RE = re.compile(r"^[A-Z]{2,}[-_][A-Z0-9_-]+$|^[A-Z]+-\d+$")
_TRAILING_CONTEXT_RE = re.compile(
    r"\s+(?:on|in|at|during|because|after|before|with|from|to|by|for)\s+.*$",
    flags=re.IGNORECASE,
)

DEFAULT_ENTITY_LABELS = [
    "Person",
    "Organization",
    "System",
    "Service",
    "API",
    "Incident",
    "Date",
    "Product",
    "Location",
    "Metric",
    "Document",
    "ErrorCode",
]

DEFAULT_RELATION_TYPES = [
    "AFFECTED",
    "CALLS",
    "CONTAINS",
    "CREATED",
    "DEPENDS_ON",
    "FOUNDED",
    "LOCATED_IN",
    "MANAGES",
    "MENTIONS",
    "OWNS",
    "PART_OF",
    "PRODUCES",
    "REPORTS_TO",
    "USES",
    "WORKS_AT",
    "RELATED_TO",
]

DEFAULT_OPENIE_RELATION_PHRASES = [
    "affected",
    "affects",
    "calls",
    "connects to",
    "contains",
    "created",
    "depends on",
    "founded",
    "located in",
    "manages",
    "mentions",
    "owns",
    "part of",
    "produces",
    "reports to",
    "runs",
    "uses",
    "works at",
]


class OpenIEBackendProtocol(Protocol):
    """Minimal protocol for Stanford/CoreNLP/OpenIE-style backends."""

    def extract_triples(self, text: str) -> list[JsonDict]:
        """Return dictionaries with subject, relation, object, and optional evidence."""


class DependencyRelationBackendProtocol(Protocol):
    """Minimal protocol for spaCy/Stanza dependency relation backends."""

    def extract_relations(self, text: str) -> list[JsonDict]:
        """Return dictionaries with source/target/relation fields and optional evidence."""


class TransformerGenerationPipelineProtocol(Protocol):
    """Minimal text-to-text generation pipeline protocol."""

    def __call__(self, text: str, **kwargs: Any) -> Any:
        """Generate relation extraction text."""


class OpenIEConfig(BaseModel):
    """Configuration for OpenIE relation extraction."""

    model_config = ConfigDict(extra="forbid")

    relation_phrases: list[str] = Field(default_factory=lambda: list(DEFAULT_OPENIE_RELATION_PHRASES))
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    schema_version: str = "v1"


class DependencyRelationPattern(BaseModel):
    """One deterministic relation pattern with source and target groups."""

    model_config = ConfigDict(extra="forbid")

    name: str
    relation_type: str
    pattern: str
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)


class RelationTypingConfig(BaseModel):
    """Configuration for GLiNER entity spans plus Ollama relation typing."""

    model_config = ConfigDict(extra="forbid")

    gliner: GlinerConfig = Field(default_factory=GlinerConfig)
    entity_labels: list[str] = Field(default_factory=lambda: list(DEFAULT_ENTITY_LABELS))
    allowed_relationship_types: list[str] = Field(default_factory=lambda: list(DEFAULT_RELATION_TYPES))
    prompt_template: str | None = None
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    schema_version: str = "v1"

    @classmethod
    def from_yaml(cls, path: str | Path) -> RelationTypingConfig:
        """Load relation typing settings from YAML."""

        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        settings_payload = payload.get("relation_typing", payload)
        return cls.model_validate(settings_payload)


class RebelConfig(BaseModel):
    """Configuration for REBEL/transformer relation extraction."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = "Babelscape/rebel-large"
    max_length: int = Field(default=256, ge=1)
    num_return_sequences: int = Field(default=1, ge=1)
    default_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    schema_version: str = "v1"


class _RelationAssembly:
    """Shared evidence and candidate construction for relation extractors."""

    def __init__(
        self,
        source: SourceRecord,
        *,
        extractor_name: str,
        technique: str,
        schema_version: str,
    ) -> None:
        self.source = source
        self.text = source.content or "\n".join(repr(row) for row in source.records)
        self.extractor_name = extractor_name
        self.technique = technique
        self.schema_version = schema_version
        self.entities: list[EntityCandidate] = []
        self.relations: list[RelationCandidate] = []
        self._entity_by_key: dict[tuple[str, str], EntityCandidate] = {}

    def add_entity(
        self,
        name: str,
        *,
        entity_type: str | None = None,
        confidence: float = 0.85,
        properties: JsonDict | None = None,
    ) -> EntityCandidate | None:
        clean_name = _clean_argument(name)
        if not clean_name:
            return None
        resolved_type = entity_type or _infer_entity_type(clean_name)
        key = (resolved_type, normalize_text(clean_name))
        if key in self._entity_by_key:
            return self._entity_by_key[key]

        span = _find_case_insensitive(self.text, clean_name)
        if span is None:
            return None
        start, end = span
        evidence = [span_evidence(self.source, start=start, end=end, text=self.text[start:end])]
        candidate = EntityCandidate.build(
            entity_type=resolved_type,
            name=clean_name,
            properties={
                "start": start,
                "end": end,
                "surface_form": self.text[start:end],
                **(properties or {}),
            },
            evidence=evidence,
            confidence=confidence,
            extractor_name=self.extractor_name,
            technique=self.technique,
            schema_version=self.schema_version,
        )
        self.entities.append(candidate)
        self._entity_by_key[key] = candidate
        return candidate

    def add_existing_entities(self, entities: list[EntityCandidate]) -> None:
        for entity in entities:
            self.entities.append(entity)
            self._entity_by_key[(entity.entity_type, normalize_text(entity.name))] = entity

    def find_entity(self, value: str) -> EntityCandidate | None:
        normalized = normalize_text(value)
        for entity in self.entities:
            if normalize_text(entity.candidate_id) == normalized or normalize_text(entity.name) == normalized:
                return entity
        return None

    def add_relation(
        self,
        *,
        source_name: str,
        relation_type: str,
        target_name: str,
        evidence_text: str,
        confidence: float,
        properties: JsonDict | None = None,
        provenance_metadata: JsonDict | None = None,
    ) -> RelationCandidate | None:
        source_entity = self.find_entity(source_name) or self.add_entity(source_name, confidence=confidence)
        target_entity = self.find_entity(target_name) or self.add_entity(target_name, confidence=confidence)
        if source_entity is None or target_entity is None:
            return None
        evidence_span = _find_case_insensitive(self.text, evidence_text, clean=False)
        if evidence_span is None:
            evidence_span = _span_covering_fragments(self.text, source_entity.name, target_entity.name)
        if evidence_span is None:
            return None
        start, end = evidence_span
        evidence = [span_evidence(self.source, start=start, end=end, text=self.text[start:end])]
        relation = RelationCandidate.build(
            source=source_entity.candidate_id,
            relation_type=_normalize_relation_type(relation_type),
            target=target_entity.candidate_id,
            properties={
                "source_name": source_entity.name,
                "target_name": target_entity.name,
                **(properties or {}),
            },
            evidence=evidence,
            confidence=confidence,
            extractor_name=self.extractor_name,
            technique=self.technique,
            schema_version=self.schema_version,
            provenance_metadata=provenance_metadata,
        )
        self.relations.append(relation)
        return relation

    def batch(self, *, metadata: JsonDict | None = None) -> ConstructionBatch:
        return ConstructionBatch(
            source_records=[self.source],
            entities=self.entities,
            relations=self.relations,
            metadata=metadata or {},
        )


class OpenIERelationExtractor:
    """Extract open-domain subject-relation-object triples with evidence spans."""

    name = "openie_relation_extractor"
    technique = "open_information_extraction"

    def __init__(
        self,
        config: OpenIEConfig | None = None,
        *,
        backend: OpenIEBackendProtocol | None = None,
    ) -> None:
        self.config = config or OpenIEConfig()
        self.backend = backend

    def extract(self, source: SourceRecord) -> ConstructionBatch:
        """Run OpenIE extraction through a backend or deterministic fallback patterns."""

        assembly = _RelationAssembly(
            source,
            extractor_name=self.name,
            technique=self.technique,
            schema_version=self.config.schema_version,
        )
        triples = self.backend.extract_triples(assembly.text) if self.backend else self._pattern_triples(assembly.text)
        for triple in triples:
            confidence = _float_value(triple.get("confidence"), 0.75)
            if confidence < self.config.min_confidence:
                continue
            subject = str(triple.get("subject", triple.get("source", "")))
            relation = str(triple.get("relation", triple.get("type", "")))
            obj = str(triple.get("object", triple.get("target", "")))
            evidence_text = str(triple.get("evidence_text") or f"{subject} {relation} {obj}")
            assembly.add_relation(
                source_name=subject,
                relation_type=relation,
                target_name=obj,
                evidence_text=evidence_text,
                confidence=confidence,
                properties={
                    "openie_relation_phrase": relation,
                    "openie_backend": self.backend.__class__.__name__ if self.backend else "pattern_fallback",
                },
            )
        return assembly.batch(
            metadata={
                "extractor": self.name,
                "backend": self.backend.__class__.__name__ if self.backend else "pattern_fallback",
                "triple_count": len(triples),
            }
        )

    def _pattern_triples(self, text: str) -> list[JsonDict]:
        relation_alternatives = "|".join(
            re.escape(phrase) for phrase in sorted(self.config.relation_phrases, key=len, reverse=True)
        )
        pattern = re.compile(
            rf"(?P<subject>{_ARGUMENT})\s+(?P<relation>{relation_alternatives})\s+"
            rf"(?P<object>{_ARGUMENT})(?=\s+(?:on|in|at|during|because|after|before|with|from|by|for)\b|[.;,!?]|$)",
            flags=re.IGNORECASE,
        )
        triples: list[JsonDict] = []
        for sentence_match in _SENTENCE_RE.finditer(text):
            sentence = sentence_match.group(0)
            offset = sentence_match.start()
            for match in pattern.finditer(sentence):
                subject = _clean_argument(match.group("subject"))
                relation = _clean_argument(match.group("relation"))
                obj = _clean_argument(match.group("object"))
                if not subject or not relation or not obj:
                    continue
                start = offset + match.start()
                end = offset + match.end()
                triples.append(
                    {
                        "subject": subject,
                        "relation": relation,
                        "object": obj,
                        "evidence_text": text[start:end],
                        "confidence": 0.78,
                    }
                )
        return triples


class DependencyPatternRelationExtractor:
    """Extract relations with deterministic dependency-style source/target patterns."""

    name = "dependency_pattern_relation_extractor"
    technique = "rule_based_dependency_patterns"

    def __init__(
        self,
        patterns: list[DependencyRelationPattern] | None = None,
        *,
        backend: DependencyRelationBackendProtocol | None = None,
        schema_version: str = "v1",
    ) -> None:
        self.patterns = patterns or default_dependency_relation_patterns()
        self.backend = backend
        self.schema_version = schema_version

    def extract(self, source: SourceRecord) -> ConstructionBatch:
        """Run dependency relation extraction through a backend or configured patterns."""

        assembly = _RelationAssembly(
            source,
            extractor_name=self.name,
            technique=self.technique,
            schema_version=self.schema_version,
        )
        if self.backend:
            raw_relations = self.backend.extract_relations(assembly.text)
            self._add_backend_relations(assembly, raw_relations)
        else:
            self._add_pattern_relations(assembly)
        return assembly.batch(
            metadata={
                "extractor": self.name,
                "backend": self.backend.__class__.__name__ if self.backend else "pattern_fallback",
                "pattern_count": len(self.patterns),
            }
        )

    def _add_backend_relations(self, assembly: _RelationAssembly, raw_relations: list[JsonDict]) -> None:
        for relation in raw_relations:
            source_name = str(relation.get("source", relation.get("subject", "")))
            target_name = str(relation.get("target", relation.get("object", "")))
            relation_type = str(relation.get("type", relation.get("relation", "RELATED_TO")))
            evidence_text = str(relation.get("evidence_text") or f"{source_name} {target_name}")
            assembly.add_relation(
                source_name=source_name,
                relation_type=relation_type,
                target_name=target_name,
                evidence_text=evidence_text,
                confidence=_float_value(relation.get("confidence"), 0.85),
                properties={"dependency_backend": self.backend.__class__.__name__ if self.backend else ""},
            )

    def _add_pattern_relations(self, assembly: _RelationAssembly) -> None:
        for relation_pattern in self.patterns:
            pattern = re.compile(relation_pattern.pattern, flags=re.IGNORECASE)
            for match in pattern.finditer(assembly.text):
                source_name = match.groupdict().get("source", "")
                target_name = match.groupdict().get("target", "")
                if not source_name or not target_name:
                    continue
                assembly.add_relation(
                    source_name=source_name,
                    relation_type=relation_pattern.relation_type,
                    target_name=target_name,
                    evidence_text=match.group(0),
                    confidence=relation_pattern.confidence,
                    properties={"dependency_pattern": relation_pattern.name},
                )


class GlinerOllamaRelationTypingExtractor:
    """Extract GLiNER entity spans, then ask Ollama to type relationships between them."""

    name = "gliner_ollama_relation_typing_extractor"
    technique = "hybrid_gliner_entities_ollama_relation_typing"

    def __init__(
        self,
        provider: OllamaProvider,
        *,
        config: RelationTypingConfig | None = None,
        entity_extractor: GlinerEntityExtractor | None = None,
        gliner_model: GlinerModelProtocol | None = None,
    ) -> None:
        self.provider = provider
        self.config = config or RelationTypingConfig()
        self.entity_extractor = entity_extractor or GlinerEntityExtractor(
            self.config.gliner,
            model=gliner_model,
        )

    def extract(self, source: SourceRecord, *, labels: list[str] | None = None) -> ConstructionBatch:
        """Run GLiNER span extraction followed by evidence-grounded LLM relation typing."""

        text = source.content or "\n".join(repr(row) for row in source.records)
        entity_labels = labels or self.config.gliner.labels or self.config.entity_labels
        entity_batch = self.entity_extractor.extract(source, labels=entity_labels)
        entity_payload = [_entity_prompt_payload(entity) for entity in entity_batch.entities]
        prompt = build_relation_typing_prompt(
            chunk_text=text,
            entities=entity_payload,
            allowed_relationship_types=self.config.allowed_relationship_types,
            prompt_template=self.config.prompt_template or DEFAULT_RELATION_TYPING_TEMPLATE,
        )
        payload = self.provider.complete_json(prompt, validator=validate_relation_typing_payload)
        assembly = _RelationAssembly(
            source,
            extractor_name=self.name,
            technique=self.technique,
            schema_version=self.config.schema_version,
        )
        assembly.add_existing_entities(entity_batch.entities)
        allowed = {_normalize_relation_type(item) for item in self.config.allowed_relationship_types}
        raw_relations = payload.get("relations", payload.get("relationships", []))
        for relation in raw_relations if isinstance(raw_relations, list) else []:
            if not isinstance(relation, dict):
                continue
            confidence = _float_value(relation.get("confidence"), 0.7)
            relation_type = _normalize_relation_type(str(relation.get("type", "RELATED_TO")))
            if confidence < self.config.min_confidence or (allowed and relation_type not in allowed):
                continue
            evidence_text = str(relation.get("evidence_text", ""))
            if not evidence_text:
                continue
            assembly.add_relation(
                source_name=str(relation.get("source", "")),
                relation_type=relation_type,
                target_name=str(relation.get("target", "")),
                evidence_text=evidence_text,
                confidence=confidence,
                properties={
                    **_dict_value(relation.get("properties")),
                    "prompt_version": RELATION_TYPING_PROMPT_VERSION,
                    "llm_model": self.provider.config.model,
                },
                provenance_metadata={
                    "entity_extractor": entity_batch.metadata.get("extractor", "gliner_entity_extractor"),
                    "relation_typer": self.provider.config.provider,
                },
            )
        return assembly.batch(
            metadata={
                "extractor": self.name,
                "entity_count": len(entity_batch.entities),
                "relation_count": len(assembly.relations),
                "llm_provider": self.provider.config.provider,
                "llm_model": self.provider.config.model,
                "prompt_version": RELATION_TYPING_PROMPT_VERSION,
            }
        )


class RebelRelationExtractor:
    """Adapter for REBEL-style generative transformer relation extraction."""

    name = "rebel_relation_extractor"
    technique = "generative_rebel_relation_extraction"

    def __init__(
        self,
        config: RebelConfig | None = None,
        *,
        pipeline: TransformerGenerationPipelineProtocol | None = None,
    ) -> None:
        self.config = config or RebelConfig()
        self.pipeline = pipeline

    def extract(self, source: SourceRecord) -> ConstructionBatch:
        """Generate REBEL triples and convert them into evidence-backed candidates."""

        assembly = _RelationAssembly(
            source,
            extractor_name=self.name,
            technique=self.technique,
            schema_version=self.config.schema_version,
        )
        generated_outputs = self._pipeline()(assembly.text, **self._generation_kwargs())
        generated_texts = _generated_texts(generated_outputs)
        triples: list[JsonDict] = []
        for generated_text in generated_texts:
            triples.extend(_parse_rebel_triples(generated_text))
        for triple in triples:
            assembly.add_relation(
                source_name=str(triple["subject"]),
                relation_type=str(triple["relation"]),
                target_name=str(triple["object"]),
                evidence_text=f"{triple['subject']} {triple['object']}",
                confidence=self.config.default_confidence,
                properties={
                    "model_name": self.config.model_name,
                    "generated_relation": triple["relation"],
                },
            )
        return assembly.batch(
            metadata={
                "extractor": self.name,
                "model_name": self.config.model_name,
                "generated_text_count": len(generated_texts),
                "triple_count": len(triples),
            }
        )

    def _pipeline(self) -> TransformerGenerationPipelineProtocol:
        if self.pipeline is not None:
            return self.pipeline
        try:
            from transformers import pipeline as transformers_pipeline
        except ImportError as exc:
            raise AdapterNotConfigured(
                "REBEL relation extraction requires transformers/torch or an injected pipeline."
            ) from exc
        pipeline_factory: Any = transformers_pipeline
        self.pipeline = pipeline_factory(
            "text2text-generation",
            model=self.config.model_name,
            tokenizer=self.config.model_name,
        )
        return self.pipeline

    def _generation_kwargs(self) -> JsonDict:
        return {
            "max_length": self.config.max_length,
            "num_return_sequences": self.config.num_return_sequences,
        }


class TransformerRelationExtractor(RebelRelationExtractor):
    """Generic transformer relation extraction adapter using REBEL-style decoding."""

    name = "transformer_relation_extractor"
    technique = "generative_transformer_relation_extraction"


def default_dependency_relation_patterns() -> list[DependencyRelationPattern]:
    """Return production-safe starter patterns for common enterprise text relations."""

    return [
        DependencyRelationPattern(
            name="reports_to",
            relation_type="REPORTS_TO",
            pattern=rf"(?P<source>{_ARGUMENT})\s+reports\s+to\s+(?P<target>{_ARGUMENT})(?=[.;,!?]|$)",
        ),
        DependencyRelationPattern(
            name="depends_on",
            relation_type="DEPENDS_ON",
            pattern=rf"(?P<source>{_ARGUMENT})\s+depends\s+on\s+(?P<target>{_ARGUMENT})(?=[.;,!?]|$)",
        ),
        DependencyRelationPattern(
            name="uses",
            relation_type="USES",
            pattern=rf"(?P<source>{_ARGUMENT})\s+uses\s+(?P<target>{_ARGUMENT})(?=[.;,!?]|$)",
        ),
        DependencyRelationPattern(
            name="affected",
            relation_type="AFFECTED",
            pattern=rf"(?P<source>{_ARGUMENT})\s+affected\s+(?P<target>{_ARGUMENT})(?=[.;,!?]|$)",
        ),
        DependencyRelationPattern(
            name="calls",
            relation_type="CALLS",
            pattern=rf"(?P<source>{_ARGUMENT})\s+calls\s+(?P<target>{_ARGUMENT})(?=[.;,!?]|$)",
        ),
    ]


def _entity_prompt_payload(entity: EntityCandidate) -> JsonDict:
    return {
        "candidate_id": entity.candidate_id,
        "name": entity.name,
        "type": entity.entity_type,
        "start": entity.properties.get("start"),
        "end": entity.properties.get("end"),
    }


def _clean_argument(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip(" \t\r\n.;,!?\"'")
    cleaned = _TRAILING_CONTEXT_RE.sub("", cleaned).strip(" \t\r\n.;,!?\"'")
    return cleaned


def _infer_entity_type(name: str) -> str:
    if _EMAIL_RE.match(name):
        return "EmailAddress"
    if _DATE_RE.match(name):
        return "Date"
    if _IDENTIFIER_RE.match(name):
        return "Identifier"
    lowered = name.casefold()
    if lowered.endswith((" api", " service", " system", " platform", " database", " store")):
        return "System"
    if any(token[:1].isupper() for token in name.split()):
        return "Entity"
    return "Concept"


def _normalize_relation_type(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return cleaned.upper() if cleaned else "RELATED_TO"


def _find_case_insensitive(text: str, fragment: str, *, clean: bool = True) -> tuple[int, int] | None:
    clean_fragment = _clean_argument(fragment) if clean else fragment.strip(" \t\r\n.;,!?\"'")
    if not clean_fragment:
        return None
    start = text.casefold().find(clean_fragment.casefold())
    if start < 0:
        return None
    return start, start + len(clean_fragment)


def _span_covering_fragments(text: str, first: str, second: str) -> tuple[int, int] | None:
    first_span = _find_case_insensitive(text, first)
    second_span = _find_case_insensitive(text, second)
    if first_span is None or second_span is None:
        return None
    return min(first_span[0], second_span[0]), max(first_span[1], second_span[1])


def _float_value(value: object, default: float) -> float:
    if isinstance(value, int | float | str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _dict_value(value: object) -> JsonDict:
    return value if isinstance(value, dict) else {}


def _generated_texts(output: object) -> list[str]:
    if isinstance(output, str):
        return [output]
    if isinstance(output, dict):
        generated = output.get("generated_text") or output.get("text")
        return [str(generated)] if generated is not None else []
    if isinstance(output, list):
        texts: list[str] = []
        for item in output:
            texts.extend(_generated_texts(item))
        return texts
    return []


def _parse_rebel_triples(generated_text: str) -> list[JsonDict]:
    text = (
        generated_text.replace("<s>", "")
        .replace("</s>", "")
        .replace("<pad>", "")
        .replace("[EOS]", "")
        .strip()
    )
    triplets: list[JsonDict] = []
    subject = ""
    obj = ""
    relation = ""
    current: str | None = None
    for token in text.split():
        if token == "<triplet>":
            if subject and obj and relation:
                triplets.append({"subject": subject.strip(), "object": obj.strip(), "relation": relation.strip()})
            subject = ""
            obj = ""
            relation = ""
            current = "subject"
            continue
        if token == "<subj>":
            current = "object"
            continue
        if token == "<obj>":
            current = "relation"
            continue
        if current == "subject":
            subject = f"{subject} {token}".strip()
        elif current == "object":
            obj = f"{obj} {token}".strip()
        elif current == "relation":
            relation = f"{relation} {token}".strip()
    if subject and obj and relation:
        triplets.append({"subject": subject.strip(), "object": obj.strip(), "relation": relation.strip()})
    return triplets


__all__ = [
    "DEFAULT_ENTITY_LABELS",
    "DEFAULT_RELATION_TYPES",
    "DependencyPatternRelationExtractor",
    "DependencyRelationBackendProtocol",
    "DependencyRelationPattern",
    "GlinerOllamaRelationTypingExtractor",
    "OpenIEBackendProtocol",
    "OpenIEConfig",
    "OpenIERelationExtractor",
    "RebelConfig",
    "RebelRelationExtractor",
    "RelationTypingConfig",
    "TransformerGenerationPipelineProtocol",
    "TransformerRelationExtractor",
    "default_dependency_relation_patterns",
]
