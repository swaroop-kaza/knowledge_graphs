"""Extensible loader adapters for optional data sources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kg_construction.domain.models import SourceRecord
from kg_construction.infrastructure.loaders.common import (
    AdapterNotConfigured,
    attach_source_evidence,
    file_location,
    normalize_rows,
)
from kg_construction.infrastructure.loaders.enterprise import (
    APIResponseLoader,
    ChatTranscriptLoader,
    EmailLoader,
    ExcelLoader,
    MarkdownLoader,
    PDFLoader,
    ReportLoader,
    SQLDatabaseLoader,
    StructuredAPILoader,
    XMLLoader,
)


class ExtensibleLoaderStub:
    """Base class for source loaders represented in the registry but not configured."""

    name = "loader_stub"
    source_type = "unknown"
    dependency_hint = "Install and configure the source-specific adapter."

    def load(self, path: str | Path) -> list[SourceRecord]:
        raise AdapterNotConfigured(
            f"{self.name} is available as an extensible adapter. {self.dependency_hint} "
            f"Input path: {path}"
        )


class DataWarehouseExportLoader(ExtensibleLoaderStub):
    name = "data_warehouse_export_loader"
    source_type = "warehouse_export"
    dependency_hint = "Use warehouse export manifests from Snowflake, BigQuery, Redshift, or Databricks."


class JSONLDLoader(ExtensibleLoaderStub):
    name = "jsonld_loader"
    source_type = "jsonld"

    def load(self, path: str | Path) -> list[SourceRecord]:
        jsonld_path = Path(path)
        payload = json.loads(jsonld_path.read_text(encoding="utf-8"))
        source = SourceRecord.from_json(
            payload,
            source_name=str(jsonld_path),
            metadata={
                "loader": self.name,
                "jsonld_context": payload.get("@context") if isinstance(payload, dict) else None,
                "jsonld_payload": payload,
            },
        ).model_copy(update={"source_type": self.source_type})
        attach_source_evidence(
            source,
            location=file_location(jsonld_path, kind="jsonld_document"),
            text=f"{jsonld_path.name}: JSON-LD document",
        )
        return [source]


class HTMLLoader(ExtensibleLoaderStub):
    name = "html_loader"
    source_type = "html"
    dependency_hint = "Use BeautifulSoup/trafilatura plus schema.org and JSON-LD extractors."


class WebPageLoader(ExtensibleLoaderStub):
    name = "webpage_loader"
    source_type = "webpage"
    dependency_hint = "Use a crawler with canonical URL, fetch policy, and boilerplate removal."


class LogLoader(ExtensibleLoaderStub):
    name = "log_loader"
    source_type = "logs"
    dependency_hint = "Use log template parsers and event timestamp normalization."


class EventPayloadLoader(ExtensibleLoaderStub):
    name = "event_payload_loader"
    source_type = "event_payload"

    def load(self, path: str | Path) -> list[SourceRecord]:
        event_path = Path(path)
        payload = json.loads(event_path.read_text(encoding="utf-8"))
        records = _event_records(payload)
        source = SourceRecord.from_rows(
            normalize_rows(records),
            source_type=self.source_type,
            source_name=str(event_path),
            metadata={
                "loader": self.name,
                "event_count": len(records),
                "json_payload": payload,
                "stream_name": payload.get("stream") if isinstance(payload, dict) else None,
            },
        )
        attach_source_evidence(
            source,
            location=file_location(event_path, kind="event_payload_file", event_count=len(records)),
            text=f"{event_path.name}: {len(records)} event payloads",
        )
        return [source]


def _event_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("events", "data", "records", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, dict)]
        return [payload]
    return [{"value": payload}]


class ScientificPaperLoader(ExtensibleLoaderStub):
    name = "scientific_paper_loader"
    source_type = "scientific_paper"
    dependency_hint = "Use GROBID/TEI, section parsing, citations, figures, and tables."


__all__ = [
    "APIResponseLoader",
    "AdapterNotConfigured",
    "ChatTranscriptLoader",
    "DataWarehouseExportLoader",
    "EmailLoader",
    "EventPayloadLoader",
    "ExcelLoader",
    "ExtensibleLoaderStub",
    "HTMLLoader",
    "JSONLDLoader",
    "LogLoader",
    "MarkdownLoader",
    "PDFLoader",
    "ReportLoader",
    "SQLDatabaseLoader",
    "ScientificPaperLoader",
    "StructuredAPILoader",
    "WebPageLoader",
    "XMLLoader",
]
