"""CSV source loader."""

from __future__ import annotations

import csv
from pathlib import Path

from kg_construction.domain.models import SourceRecord
from kg_construction.infrastructure.loaders.common import attach_source_evidence, file_location


class CsvLoader:
    """Load CSV rows into one source record."""

    name = "csv_loader"
    source_type = "csv"

    def load(self, path: str | Path) -> list[SourceRecord]:
        csv_path = Path(path)
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
        source = SourceRecord.from_rows(
            rows,
            source_type=self.source_type,
            source_name=str(csv_path),
            metadata={"loader": self.name, "row_count": len(rows)},
        )
        attach_source_evidence(
            source,
            location=file_location(csv_path, kind="csv_file", row_count=len(rows)),
            text=f"{csv_path.name}: {len(rows)} rows",
        )
        return [source]
