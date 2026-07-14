"""Dictionary and controlled-vocabulary extraction."""

from __future__ import annotations

import re

from kg_construction.domain.evidence import span_evidence
from kg_construction.domain.models import ConstructionBatch, EntityCandidate, SourceRecord


class GazetteerEntityExtractor:
    """Extract entities from configured label -> term dictionaries."""

    name = "gazetteer_entity_extractor"
    technique = "rule_based_gazetteer"

    def __init__(self, gazetteer: dict[str, list[str]], *, confidence: float = 0.95) -> None:
        self.gazetteer = gazetteer
        self.confidence = confidence

    def extract(self, source: SourceRecord) -> ConstructionBatch:
        text = source.content or "\n".join(repr(row) for row in source.records)
        entities: list[EntityCandidate] = []
        for entity_type, terms in self.gazetteer.items():
            for term in terms:
                pattern = re.compile(rf"\b{re.escape(term)}\b", flags=re.IGNORECASE)
                for match in pattern.finditer(text):
                    evidence = [
                        span_evidence(source, start=match.start(), end=match.end(), text=match.group(0))
                    ]
                    entities.append(
                        EntityCandidate.build(
                            entity_type=entity_type,
                            name=match.group(0),
                            properties={"gazetteer_term": term, "start": match.start(), "end": match.end()},
                            evidence=evidence,
                            confidence=self.confidence,
                            extractor_name=self.name,
                            technique=self.technique,
                        )
                    )
        return ConstructionBatch(source_records=[source], entities=entities, metadata={"extractor": self.name})


__all__ = ["GazetteerEntityExtractor"]
