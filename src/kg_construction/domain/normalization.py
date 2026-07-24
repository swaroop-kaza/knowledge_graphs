"""Normalize source values to their declared datatype.

Structured sources disagree on how they spell the same value: integers arrive as
strings, dates in mixed formats, timestamps with or without a time zone. Mapping
must normalize to one canonical form so that identity, deduplication, and
re-ingestion are stable. Naive timestamps are anchored to UTC; dates and
datetimes are emitted in ISO 8601. Values that cannot be parsed are returned
unchanged as strings, so normalization never loses data.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any


def _to_int(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            return str(value)


def _to_number(value: Any) -> Any:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return str(value)


def _to_boolean(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if token in {"true", "t", "yes", "y", "1"}:
        return True
    if token in {"false", "f", "no", "n", "0"}:
        return False
    return str(value)


def _to_date(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        return text


def _to_datetime(value: Any) -> Any:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return str(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


def normalize_value(value: Any, datatype: str | None) -> Any:
    """Return ``value`` normalized to ``datatype`` (canonical, stable form)."""

    if value is None:
        return None
    if datatype == "integer":
        return _to_int(value)
    if datatype == "number":
        return _to_number(value)
    if datatype == "boolean":
        return _to_boolean(value)
    if datatype == "date":
        return _to_date(value)
    if datatype == "datetime":
        return _to_datetime(value)
    if datatype in (None, "string", "object", "array"):
        return value if isinstance(value, str) else str(value)
    return value


__all__ = ["normalize_value"]
