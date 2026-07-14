"""Rule-based regex extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from re import Pattern

from kg_construction.domain.evidence import span_evidence
from kg_construction.domain.models import ConstructionBatch, EntityCandidate, SourceRecord


@dataclass(frozen=True, slots=True)
class RegexPattern:
    """A compiled regex pattern and output entity type."""

    entity_type: str
    pattern: Pattern[str]
    confidence: float = 0.98


DEFAULT_PATTERNS: tuple[RegexPattern, ...] = (
    RegexPattern("EmailAddress", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    RegexPattern("URL", re.compile(r"https?://[^\s)>\"]+")),
    RegexPattern("Date", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    RegexPattern("Identifier", re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")),
    RegexPattern("Version", re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b")),
)


class RegexEntityExtractor:
    """Extract precise spans such as emails, URLs, dates, versions, and IDs."""

    name = "regex_entity_extractor"
    technique = "rule_based_regex"

    def __init__(self, patterns: tuple[RegexPattern, ...] = DEFAULT_PATTERNS) -> None:
        self.patterns = patterns

    def extract(self, source: SourceRecord) -> ConstructionBatch:
        text = source.content or "\n".join(repr(row) for row in source.records)
        entities: list[EntityCandidate] = []
        seen: set[tuple[str, int, int]] = set()

        for pattern in self.patterns:
            for match in pattern.pattern.finditer(text):
                key = (pattern.entity_type, match.start(), match.end())
                if key in seen:
                    continue
                seen.add(key)
                evidence = [
                    span_evidence(source, start=match.start(), end=match.end(), text=match.group(0))
                ]
                entities.append(
                    EntityCandidate.build(
                        entity_type=pattern.entity_type,
                        name=match.group(0),
                        properties={"start": match.start(), "end": match.end()},
                        evidence=evidence,
                        confidence=pattern.confidence,
                        extractor_name=self.name,
                        technique=self.technique,
                    )
                )

        return ConstructionBatch(source_records=[source], entities=entities, metadata={"extractor": self.name})


__all__ = ["DEFAULT_PATTERNS", "RegexEntityExtractor", "RegexPattern"]
