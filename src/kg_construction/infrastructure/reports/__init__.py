"""Report renderers for technique evaluation."""

from __future__ import annotations

from kg_construction.infrastructure.reports.comparison_matrix import build_comparison_matrix
from kg_construction.infrastructure.reports.json_report import JsonReportWriter
from kg_construction.infrastructure.reports.markdown_report import MarkdownReportWriter

__all__ = ["JsonReportWriter", "MarkdownReportWriter", "build_comparison_matrix"]
