"""Incremental sync and watermark tracking."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.domain.models import SourceRecord, utc_now

JsonDict = dict[str, Any]


class WatermarkState(BaseModel):
    """Stored high-watermark for one source key."""

    model_config = ConfigDict(extra="forbid")

    source_key: str
    high_watermark: str
    source_id: str
    processed_at: str = Field(default_factory=lambda: utc_now().isoformat())
    metadata: JsonDict = Field(default_factory=dict)


class IncrementalSyncResult(BaseModel):
    """Result of applying watermark filtering to source records."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    eligible_sources: list[SourceRecord] = Field(default_factory=list)
    skipped_sources: list[SourceRecord] = Field(default_factory=list)
    watermark_values: dict[str, str] = Field(default_factory=dict)


class WatermarkStore:
    """JSON-backed watermark store for deterministic local and CI runs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._states = self._load()

    def get(self, source_key: str) -> WatermarkState | None:
        return self._states.get(source_key)

    def update(
        self,
        *,
        source_key: str,
        high_watermark: str,
        source_id: str,
        metadata: JsonDict | None = None,
    ) -> WatermarkState:
        state = WatermarkState(
            source_key=source_key,
            high_watermark=high_watermark,
            source_id=source_id,
            metadata=metadata or {},
        )
        self._states[source_key] = state
        self.flush()
        return state

    def flush(self) -> None:
        payload = {
            key: value.model_dump(mode="json")
            for key, value in sorted(self._states.items())
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _load(self) -> dict[str, WatermarkState]:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8") or "{}")
        return {key: WatermarkState.model_validate(value) for key, value in payload.items()}


class IncrementalSyncPolicy:
    """Filter sources by source-level watermark metadata."""

    def __init__(
        self,
        *,
        watermark_field: str = "updated_at",
        fallback_to_source_id: bool = True,
    ) -> None:
        self.watermark_field = watermark_field
        self.fallback_to_source_id = fallback_to_source_id

    def apply(self, sources: list[SourceRecord], store: WatermarkStore) -> IncrementalSyncResult:
        eligible: list[SourceRecord] = []
        skipped: list[SourceRecord] = []
        values: dict[str, str] = {}
        for source in sources:
            source_key = self.source_key(source)
            value = self.watermark_value(source)
            values[source_key] = value
            current_state = store.get(source_key)
            if current_state is None or _compare_watermarks(value, current_state.high_watermark) > 0:
                eligible.append(source)
            else:
                skipped.append(source)
        return IncrementalSyncResult(
            eligible_sources=eligible,
            skipped_sources=skipped,
            watermark_values=values,
        )

    def commit(self, sources: list[SourceRecord], store: WatermarkStore) -> None:
        for source in sources:
            store.update(
                source_key=self.source_key(source),
                high_watermark=self.watermark_value(source),
                source_id=source.source_id,
                metadata={
                    "source_type": source.source_type,
                    "source_name": source.source_name,
                    "watermark_field": self.watermark_field,
                },
            )

    def source_key(self, source: SourceRecord) -> str:
        return f"{source.source_type}:{source.source_name}"

    def watermark_value(self, source: SourceRecord) -> str:
        value = source.metadata.get(self.watermark_field)
        if value not in (None, ""):
            return str(value)
        if self.fallback_to_source_id:
            return source.source_id
        return "0"


def _compare_watermarks(left: str, right: str) -> int:
    left_dt = _parse_datetime(left)
    right_dt = _parse_datetime(right)
    if left_dt is not None and right_dt is not None:
        return (left_dt > right_dt) - (left_dt < right_dt)
    return (left > right) - (left < right)


def _parse_datetime(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


__all__ = [
    "IncrementalSyncPolicy",
    "IncrementalSyncResult",
    "WatermarkState",
    "WatermarkStore",
]
