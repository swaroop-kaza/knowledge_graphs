"""JSON report writer."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


class JsonReportWriter:
    """Write summary-first evaluation JSON without embedding source-derived graphs."""

    name = "json_report_writer"

    _SENSITIVE_CANDIDATE_FIELDS = {
        "batch",
        "property_graph",
        "rdf_graph",
        "evidence_graph",
    }

    def __init__(self, *, include_sensitive_artifacts: bool = False) -> None:
        self.include_sensitive_artifacts = include_sensitive_artifacts

    def payload(self, report: Any) -> Any:
        """Return the JSON payload, excluding source-derived artifacts by default."""

        payload = (
            report.model_dump(mode="json")
            if hasattr(report, "model_dump")
            else deepcopy(report)
        )
        if not isinstance(payload, dict) or self.include_sensitive_artifacts:
            return payload
        candidates = payload.get("candidates", [])
        if isinstance(candidates, list):
            for candidate in candidates:
                if isinstance(candidate, dict):
                    for field in self._SENSITIVE_CANDIDATE_FIELDS:
                        candidate.pop(field, None)
        input_path = payload.get("input_path")
        if isinstance(input_path, str):
            payload["input_path"] = _portable_path(input_path)
        run_metadata = payload.get("run_metadata")
        if isinstance(run_metadata, dict):
            gold_path = run_metadata.get("gold_dataset_path")
            if isinstance(gold_path, str):
                run_metadata["gold_dataset_path"] = _portable_path(gold_path)
        payload["publication_serialization"] = {
            "mode": "summary_only",
            "source_derived_artifacts_included": False,
            "notice": (
                "Candidate batches and graphs are omitted because they can contain source "
                "content or personal data. Inspect remaining metadata before publication."
            ),
        }
        return payload

    def write(self, report: Any, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.payload(report)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _portable_path(value: str) -> str:
    path = Path(value)
    return path.name if path.is_absolute() or ".." in path.parts else path.as_posix()


__all__ = ["JsonReportWriter"]
