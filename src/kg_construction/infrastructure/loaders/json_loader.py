"""JSON source loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kg_construction.domain.models import SourceRecord
from kg_construction.infrastructure.loaders.common import attach_source_evidence, file_location


class JsonLoader:
    """Load JSON files into source records."""

    name = "json_loader"
    source_type = "json"

    def load(self, path: str | Path) -> list[SourceRecord]:
        json_path = Path(path)
        with json_path.open(encoding="utf-8") as handle:
            payload: Any = json.load(handle)
        source = SourceRecord.from_json(
            payload,
            source_name=str(json_path),
            metadata={"loader": self.name},
        )
        attach_source_evidence(
            source,
            location=file_location(json_path, kind="json_file"),
            text=json_path.name,
            metadata={
                "payload_type": type(payload).__name__,
                "record_count": len(payload) if isinstance(payload, list) else None,
            },
        )
        return [source]
