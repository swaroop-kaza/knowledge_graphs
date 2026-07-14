"""JSON report writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonReportWriter:
    """Write evaluation reports as reproducible JSON."""

    name = "json_report_writer"

    def write(self, report: Any, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = report.model_dump(mode="json") if hasattr(report, "model_dump") else report
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


__all__ = ["JsonReportWriter"]
