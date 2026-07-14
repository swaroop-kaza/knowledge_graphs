"""GLiNER zero-shot entity extraction."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.domain.evidence import span_evidence
from kg_construction.domain.models import ConstructionBatch, EntityCandidate, SourceRecord
from kg_construction.infrastructure.llm.label_proposal import normalize_label_name
from kg_construction.infrastructure.loaders.stubs import AdapterNotConfigured


class GlinerModelProtocol(Protocol):
    """Minimal GLiNER model interface used by the extractor."""

    def predict_entities(
        self,
        text: str,
        labels: list[str],
        *,
        threshold: float = 0.5,
        flat_ner: bool = True,
    ) -> list[dict[str, Any]]:
        """Return GLiNER entity dictionaries."""


class GlinerConfig(BaseModel):
    """Configuration for GLiNER inference."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = "gliner-community/gliner_small-v2.5"
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    flat_ner: bool = True
    labels: list[str] = Field(default_factory=list)
    schema_version: str = "v1"


class GlinerEntityExtractor:
    """Extract entity spans with GLiNER using dynamic zero-shot labels."""

    name = "gliner_entity_extractor"
    technique = "zero_shot_gliner"

    def __init__(
        self,
        config: GlinerConfig | None = None,
        *,
        model: GlinerModelProtocol | None = None,
    ) -> None:
        self.config = config or GlinerConfig()
        self._model = model

    def extract(self, source: SourceRecord, *, labels: list[str] | None = None) -> ConstructionBatch:
        """Run GLiNER over a source record and return evidence-backed entities."""

        text = source.content or "\n".join(repr(row) for row in source.records)
        effective_labels = [normalize_label_name(label) for label in (labels or self.config.labels)]
        effective_labels = [label for label in dict.fromkeys(effective_labels) if label]
        if not effective_labels:
            raise AdapterNotConfigured("GLiNER extraction requires at least one entity label.")

        model = self._load_model()
        raw_entities = model.predict_entities(
            text,
            effective_labels,
            threshold=self.config.threshold,
            flat_ner=self.config.flat_ner,
        )

        entities: list[EntityCandidate] = []
        seen: set[tuple[str, int, int]] = set()
        for raw_entity in raw_entities:
            label = normalize_label_name(str(raw_entity.get("label", raw_entity.get("entity", ""))))
            entity_text = str(raw_entity.get("text", raw_entity.get("word", "")))
            start = int(raw_entity.get("start", -1))
            end = int(raw_entity.get("end", -1))
            if not label or not entity_text or start < 0 or end <= start:
                continue
            key = (label, start, end)
            if key in seen:
                continue
            seen.add(key)
            evidence = [span_evidence(source, start=start, end=end, text=entity_text)]
            score = raw_entity.get("score", raw_entity.get("confidence", 0.7))
            entities.append(
                EntityCandidate.build(
                    entity_type=label,
                    name=entity_text,
                    properties={
                        "start": start,
                        "end": end,
                        "gliner_label": label,
                        "gliner_model": self.config.model_name,
                    },
                    evidence=evidence,
                    confidence=float(score) if isinstance(score, int | float | str) else 0.7,
                    extractor_name=self.name,
                    technique=self.technique,
                    schema_version=self.config.schema_version,
                    provenance_metadata={
                        "model_name": self.config.model_name,
                        "threshold": self.config.threshold,
                    },
                )
            )

        return ConstructionBatch(
            source_records=[source],
            entities=entities,
            metadata={
                "extractor": self.name,
                "model_name": self.config.model_name,
                "threshold": self.config.threshold,
                "labels": effective_labels,
            },
        )

    def _load_model(self) -> GlinerModelProtocol:
        if self._model is not None:
            return self._model
        try:
            from gliner import GLiNER
        except ImportError as exc:
            raise AdapterNotConfigured(
                "GLiNER is not installed. Install the gliner extra or pass an object "
                "with predict_entities(text, labels, threshold=..., flat_ner=...)."
            ) from exc
        self._model = GLiNER.from_pretrained(self.config.model_name)
        return self._model


__all__ = ["GlinerConfig", "GlinerEntityExtractor", "GlinerModelProtocol"]
