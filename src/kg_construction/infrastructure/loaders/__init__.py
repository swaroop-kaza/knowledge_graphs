"""Source loaders."""

from __future__ import annotations

from kg_construction.infrastructure.loaders.csv_loader import CsvLoader
from kg_construction.infrastructure.loaders.enterprise import (
    APIResponseLoader,
    ChatTranscriptLoader,
    EmailLoader,
    ExcelLoader,
    HTMLLoader,
    LogLoader,
    MarkdownLoader,
    PDFLoader,
    ReportLoader,
    ScientificPaperLoader,
    SQLDatabaseLoader,
    StructuredAPILoader,
    WebPageLoader,
    XMLLoader,
)
from kg_construction.infrastructure.loaders.json_loader import JsonLoader
from kg_construction.infrastructure.loaders.stubs import (
    DataWarehouseExportLoader,
    EventPayloadLoader,
    JSONLDLoader,
)
from kg_construction.infrastructure.loaders.text_loader import TextLoader

__all__ = [
    "APIResponseLoader",
    "ChatTranscriptLoader",
    "CsvLoader",
    "DataWarehouseExportLoader",
    "EmailLoader",
    "EventPayloadLoader",
    "ExcelLoader",
    "HTMLLoader",
    "JSONLDLoader",
    "JsonLoader",
    "LogLoader",
    "MarkdownLoader",
    "PDFLoader",
    "ReportLoader",
    "SQLDatabaseLoader",
    "ScientificPaperLoader",
    "StructuredAPILoader",
    "TextLoader",
    "WebPageLoader",
    "XMLLoader",
]
