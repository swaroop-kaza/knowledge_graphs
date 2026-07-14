"""Text source loader."""

from __future__ import annotations

from pathlib import Path

from kg_construction.domain.models import SourceRecord
from kg_construction.infrastructure.loaders.common import attach_source_evidence, file_location


class TextLoader:
    """Load plain text or markdown into a source record."""

    name = "text_loader"
    source_type = "text"

    def load(self, path: str | Path) -> list[SourceRecord]:
        text_path = Path(path)
        text = text_path.read_text(encoding="utf-8")
        source_type = "markdown" if text_path.suffix.lower() in {".md", ".markdown"} else self.source_type
        source = SourceRecord.from_text(
            text,
            source_type=source_type,
            source_name=str(text_path),
            metadata={"loader": self.name, "suffix": text_path.suffix.lower()},
        )
        attach_source_evidence(
            source,
            location=file_location(text_path, kind=f"{source_type}_file"),
            text=text_path.name,
            metadata={"character_count": len(text)},
        )
        return [source]
