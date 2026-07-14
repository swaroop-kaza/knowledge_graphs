"""Dataset profiling for technique selection."""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.domain.models import SourceRecord
from kg_construction.infrastructure.mapping.direct_table_mapper import profile_records

DatasetKind = Literal["structured", "semi_structured", "unstructured", "mixed", "unknown"]

_ENTITY_HINT_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
_RELATION_HINT_RE = re.compile(r"\b(works at|owns|uses|reports to|part of|located in|causes|depends on)\b", re.I)


def clamp(value: float) -> float:
    """Clamp a score to the 0..1 range."""

    return max(0.0, min(1.0, value))


class DatasetProfile(BaseModel):
    """Normalized profile used for applicability and scoring."""

    model_config = ConfigDict(extra="forbid")

    data_type: DatasetKind = "unknown"
    file_format: str = "unknown"
    source_count: int = 0
    record_count: int = 0
    character_count: int = 0
    has_schema: bool = False
    primary_keys: list[str] = Field(default_factory=list)
    foreign_keys: list[tuple[str, str]] = Field(default_factory=list)
    column_types: dict[str, str] = Field(default_factory=dict)
    null_rates: dict[str, float] = Field(default_factory=dict)
    duplicate_rate: float = 0.0
    text_density: float = 0.0
    table_density: float = 0.0
    nestedness_score: float = 0.0
    schema_drift_score: float = 0.0
    document_length: int = 0
    chunkability_score: float = 0.0
    language: str = "unknown"
    entity_density_score: float = 0.0
    relation_density_score: float = 0.0
    ocr_layout_complexity_score: float = 0.0
    ontology_alignment_score: float = 0.0
    source_reliability_score: float = 0.8
    update_frequency_score: float = 0.0
    expected_graph_size: int = 0
    structuredness_score: float = 0.0
    schema_quality_score: float = 0.0
    text_complexity_score: float = 0.0
    evidence_quality_score: float = 0.0
    data_volume_score: float = 0.0
    warnings: list[str] = Field(default_factory=list)

    def score_payload(self) -> dict[str, float]:
        """Return the score vector requested by the framework document."""

        return {
            "structuredness_score": self.structuredness_score,
            "schema_quality_score": self.schema_quality_score,
            "text_complexity_score": self.text_complexity_score,
            "entity_density_score": self.entity_density_score,
            "relation_density_score": self.relation_density_score,
            "nestedness_score": self.nestedness_score,
            "evidence_quality_score": self.evidence_quality_score,
            "ontology_alignment_score": self.ontology_alignment_score,
            "data_volume_score": self.data_volume_score,
            "update_frequency_score": self.update_frequency_score,
        }


class DatasetProfiler:
    """Profile source records or files before selecting construction techniques."""

    name = "dataset_profiler"

    def profile_sources(self, sources: list[SourceRecord], *, input_path: str | None = None) -> DatasetProfile:
        if not sources:
            return DatasetProfile(warnings=["No sources were provided."])

        source_types = {source.source_type for source in sources}
        records = [row for source in sources for row in source.records]
        texts = [source.content or "" for source in sources]
        combined_text = "\n".join(text for text in texts if text)
        character_count = len(combined_text) + sum(len(repr(row)) for row in records)
        file_format = self._file_format(input_path, source_types)
        profile = profile_records(records) if records else {}
        column_types = {
            str(column): str(column_type)
            for column, column_type in profile.get("types", {}).items()
        }
        null_rates = self._null_rates(profile, max(1, len(records)))
        primary_keys = self._primary_keys(profile)
        foreign_keys = self._foreign_keys(records, primary_keys)
        nestedness_score = self._nestedness(records, combined_text)
        duplicate_rate = self._duplicate_rate(records, combined_text)
        text_density = self._text_density(records, combined_text)
        table_density = clamp(len(records) / max(1, len(records) + len(texts)))
        structuredness_score = self._structuredness(records, source_types)
        schema_quality_score = self._schema_quality(records, primary_keys, foreign_keys, null_rates)
        entity_density_score = self._entity_density(combined_text, records)
        relation_density_score = self._relation_density(combined_text, records, foreign_keys)
        text_complexity_score = self._text_complexity(combined_text, nestedness_score)
        data_volume_score = clamp((len(records) + max(1, character_count // 1000)) / 1000)
        document_length = max((len(text) for text in texts), default=0)
        data_type = self._data_type(structuredness_score, text_density, nestedness_score, source_types)

        return DatasetProfile(
            data_type=data_type,
            file_format=file_format,
            source_count=len(sources),
            record_count=len(records),
            character_count=character_count,
            has_schema=bool(column_types),
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            column_types=column_types,
            null_rates=null_rates,
            duplicate_rate=duplicate_rate,
            text_density=text_density,
            table_density=table_density,
            nestedness_score=nestedness_score,
            schema_drift_score=0.0,
            document_length=document_length,
            chunkability_score=clamp(document_length / 1500),
            language="en" if combined_text else "unknown",
            entity_density_score=entity_density_score,
            relation_density_score=relation_density_score,
            ocr_layout_complexity_score=1.0 if file_format in {"pdf", "image"} else 0.0,
            ontology_alignment_score=1.0 if file_format in {"rdf", "ttl", "jsonld"} else 0.0,
            source_reliability_score=0.9 if file_format in {"csv", "json", "txt"} else 0.75,
            update_frequency_score=0.0,
            expected_graph_size=max(len(records), int(entity_density_score * 20), len(foreign_keys)),
            structuredness_score=structuredness_score,
            schema_quality_score=schema_quality_score,
            text_complexity_score=text_complexity_score,
            evidence_quality_score=0.95 if sources else 0.0,
            data_volume_score=data_volume_score,
            warnings=self._warnings(records, combined_text, primary_keys),
        )

    def profile_path(self, path: str | Path, *, source_type: str | None = None) -> DatasetProfile:
        data_path = Path(path)
        inferred = source_type or data_path.suffix.lower().lstrip(".") or "unknown"
        if inferred == "csv":
            from kg_construction.infrastructure.loaders.csv_loader import CsvLoader

            return self.profile_sources(CsvLoader().load(data_path), input_path=str(data_path))
        if inferred == "json":
            from kg_construction.infrastructure.loaders.json_loader import JsonLoader

            return self.profile_sources(JsonLoader().load(data_path), input_path=str(data_path))
        if inferred in {"xlsx", "xlsm"}:
            from kg_construction.infrastructure.loaders import ExcelLoader

            return self.profile_sources(ExcelLoader().load(data_path), input_path=str(data_path))
        if inferred in {"xml"}:
            from kg_construction.infrastructure.loaders import XMLLoader

            return self.profile_sources(XMLLoader().load(data_path), input_path=str(data_path))
        if inferred in {"pdf"}:
            from kg_construction.infrastructure.loaders import PDFLoader

            return self.profile_sources(PDFLoader().load(data_path), input_path=str(data_path))
        if inferred in {"eml", "email"}:
            from kg_construction.infrastructure.loaders import EmailLoader

            return self.profile_sources(EmailLoader().load(data_path), input_path=str(data_path))
        if inferred in {"chat", "transcript"}:
            from kg_construction.infrastructure.loaders import ChatTranscriptLoader

            return self.profile_sources(ChatTranscriptLoader().load(data_path), input_path=str(data_path))
        if inferred in {"report"}:
            from kg_construction.infrastructure.loaders import ReportLoader

            return self.profile_sources(ReportLoader().load(data_path), input_path=str(data_path))
        if inferred in {"md", "markdown"}:
            from kg_construction.infrastructure.loaders import MarkdownLoader

            return self.profile_sources(MarkdownLoader().load(data_path), input_path=str(data_path))
        if inferred in {"html", "htm"}:
            from kg_construction.infrastructure.loaders import HTMLLoader

            return self.profile_sources(HTMLLoader().load(data_path), input_path=str(data_path))
        if inferred in {"log"}:
            from kg_construction.infrastructure.loaders import LogLoader

            return self.profile_sources(LogLoader().load(data_path), input_path=str(data_path))
        if inferred in {"tei", "jats"} or source_type == "scientific_paper":
            from kg_construction.infrastructure.loaders import ScientificPaperLoader

            return self.profile_sources(ScientificPaperLoader().load(data_path), input_path=str(data_path))
        from kg_construction.infrastructure.loaders.text_loader import TextLoader

        return self.profile_sources(TextLoader().load(data_path), input_path=str(data_path))

    @staticmethod
    def _file_format(input_path: str | None, source_types: set[str]) -> str:
        if input_path:
            suffix = Path(input_path).suffix.lower().lstrip(".")
            if suffix:
                return "txt" if suffix == "text" else suffix
        return sorted(source_types)[0] if source_types else "unknown"

    @staticmethod
    def _null_rates(profile: dict[str, Any], row_count: int) -> dict[str, float]:
        null_counts = profile.get("null_counts", {})
        return {str(column): count / row_count for column, count in null_counts.items()}

    @staticmethod
    def _primary_keys(profile: dict[str, Any]) -> list[str]:
        columns = profile.get("columns", [])
        row_count = int(profile.get("row_count", 0))
        distinct_counts = profile.get("distinct_counts", {})
        likely = [
            str(column)
            for column in columns
            if str(column).lower() in {"id", "uuid", "key"}
            or (row_count > 0 and distinct_counts.get(column) == row_count)
        ]
        return likely[:3]

    @staticmethod
    def _foreign_keys(records: list[dict[str, Any]], primary_keys: list[str]) -> list[tuple[str, str]]:
        if not records or not primary_keys:
            return []
        values_by_key = {
            key: {str(row.get(key)) for row in records if row.get(key) not in (None, "")}
            for key in primary_keys
        }
        foreign_keys: list[tuple[str, str]] = []
        for column in sorted({key for row in records for key in row}):
            if not column.endswith("_id") or column in primary_keys:
                continue
            values = {str(row.get(column)) for row in records if row.get(column) not in (None, "")}
            for primary_key, primary_values in values_by_key.items():
                if values and values <= primary_values:
                    foreign_keys.append((column, primary_key))
        return foreign_keys

    @staticmethod
    def _nestedness(records: list[dict[str, Any]], text: str) -> float:
        nested_values = 0
        total_values = 0
        for row in records:
            for value in row.values():
                total_values += 1
                if isinstance(value, dict | list):
                    nested_values += 1
        if total_values:
            return clamp(nested_values / total_values)
        if text.strip().startswith(("{", "[")):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                return 0.0
            return 0.7 if isinstance(payload, dict | list) else 0.0
        return 0.0

    @staticmethod
    def _duplicate_rate(records: list[dict[str, Any]], text: str) -> float:
        if records:
            keys = [repr(sorted(row.items())) for row in records]
            return 1.0 - (len(set(keys)) / len(keys))
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return 0.0 if not lines else 1.0 - (len(set(lines)) / len(lines))

    @staticmethod
    def _text_density(records: list[dict[str, Any]], text: str) -> float:
        row_chars = sum(len(repr(row)) for row in records)
        text_chars = len(text)
        return text_chars / max(1, text_chars + row_chars)

    @staticmethod
    def _structuredness(records: list[dict[str, Any]], source_types: set[str]) -> float:
        if records:
            return 0.95 if {"csv", "excel", "sql"} & source_types else 0.75
        return 0.2

    @staticmethod
    def _schema_quality(
        records: list[dict[str, Any]],
        primary_keys: list[str],
        foreign_keys: list[tuple[str, str]],
        null_rates: dict[str, float],
    ) -> float:
        if not records:
            return 0.1
        key_score = 0.4 if primary_keys else 0.0
        fk_score = 0.2 if foreign_keys else 0.0
        null_score = 0.4 * (1.0 - mean(null_rates.values())) if null_rates else 0.3
        return clamp(key_score + fk_score + null_score)

    @staticmethod
    def _entity_density(text: str, records: list[dict[str, Any]]) -> float:
        if records:
            meaningful_columns = [
                column
                for row in records
                for column, value in row.items()
                if value not in (None, "") and str(column).lower() not in {"id", "uuid"}
            ]
            return clamp(len(meaningful_columns) / max(1, len(records) * 5))
        tokens = max(1, len(text.split()))
        return clamp(len(_ENTITY_HINT_RE.findall(text)) / tokens)

    @staticmethod
    def _relation_density(
        text: str,
        records: list[dict[str, Any]],
        foreign_keys: list[tuple[str, str]],
    ) -> float:
        if records:
            return clamp(len(foreign_keys) / max(1, len(records)))
        tokens = max(1, len(text.split()))
        return clamp((len(_RELATION_HINT_RE.findall(text)) * 5) / tokens)

    @staticmethod
    def _text_complexity(text: str, nestedness_score: float) -> float:
        if not text:
            return nestedness_score * 0.3
        words = text.split()
        avg_word_len = mean(len(word) for word in words) if words else 0.0
        sentence_count = max(1, text.count(".") + text.count("\n"))
        words_per_sentence = len(words) / sentence_count
        return clamp((avg_word_len / 12) * 0.4 + (words_per_sentence / 40) * 0.4 + nestedness_score * 0.2)

    @staticmethod
    def _data_type(
        structuredness_score: float,
        text_density: float,
        nestedness_score: float,
        source_types: set[str],
    ) -> DatasetKind:
        if len(source_types) > 1:
            return "mixed"
        if structuredness_score >= 0.7 and text_density < 0.6:
            return "structured"
        if nestedness_score >= 0.2:
            return "semi_structured"
        if text_density >= 0.6:
            return "unstructured"
        return "unknown"

    @staticmethod
    def _warnings(records: list[dict[str, Any]], text: str, primary_keys: list[str]) -> list[str]:
        warnings: list[str] = []
        if records and not primary_keys:
            warnings.append("No stable primary key was detected; deterministic IDs will use row content.")
        if text and len(text) > 4000:
            warnings.append("Long document detected; chunking and latency-aware extraction are recommended.")
        return warnings


__all__ = ["DatasetKind", "DatasetProfile", "DatasetProfiler", "clamp"]
