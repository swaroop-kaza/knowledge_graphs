"""Deterministic identity primitives for structured graph construction.

Keys need stricter treatment than ordinary values.  In particular, an identifier
that merely *looks* numeric remains lexical unless its declaration explicitly
assigns a numeric datatype.  These helpers centralise collision-safe composite
serialization and the two uniqueness checks required before graph materialisation:

* raw source keys must be unique; and
* their datatype-normalised forms must also be unique.

The second check catches lossy declarations such as ``"001"`` and ``"1"`` both
being declared as integers.  Construction fails before either row can be merged.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from kg_construction.domain.models import canonical_json
from kg_construction.domain.normalization import normalize_value

JsonDict = dict[str, Any]
TargetKeyLookup = tuple[str, str, tuple[str, ...], str]
TargetKeyIndex = dict[TargetKeyLookup, str]

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_key_value(value: object, datatype: str | None) -> object:
    """Return the canonical key component for ``datatype``.

    Lexical identifiers receive Unicode/edge-whitespace normalisation while
    retaining case and leading zeros.  Numeric coercion happens only when the
    resolved declaration explicitly carries a numeric datatype.
    """

    normalized = normalize_value(value, datatype)
    if datatype in (None, "string"):
        text = unicodedata.normalize("NFKC", str(normalized))
        return _WHITESPACE_RE.sub(" ", text).strip()
    return normalized


def canonical_key(values: Sequence[object], datatypes: Sequence[str | None]) -> str:
    """Serialize a composite key without delimiter collisions."""

    if len(values) != len(datatypes):
        raise ValueError(
            f"key has {len(values)} value(s) but {len(datatypes)} datatype declaration(s)"
        )
    return canonical_json(
        [
            normalize_key_value(value, datatype)
            for value, datatype in zip(values, datatypes, strict=True)
        ]
    )


def validate_unique_key(
    records: Sequence[Mapping[str, object]],
    *,
    columns: Sequence[str],
    datatypes: Sequence[str | None],
    key_name: str,
) -> dict[str, int]:
    """Validate a declared key and return canonical key -> source row index.

    Missing components, duplicate raw tuples, and normalisation collisions are
    all hard failures.  This prevents graph identity from silently coalescing
    distinct source rows.
    """

    if not columns:
        raise ValueError(f"{key_name} must contain at least one column")
    if len(columns) != len(datatypes):
        raise ValueError(
            f"{key_name} has {len(columns)} column(s) but {len(datatypes)} datatype(s)"
        )

    raw_rows: dict[str, int] = {}
    canonical_rows: dict[str, tuple[int, str]] = {}
    result: dict[str, int] = {}
    for row_index, row in enumerate(records):
        missing = [
            column for column in columns if row.get(column) in (None, "")
        ]
        if missing:
            raise ValueError(
                f"{key_name} is missing column(s) {missing!r} on row {row_index}"
            )
        values = [row[column] for column in columns]
        raw = canonical_json(values)
        previous_raw = raw_rows.get(raw)
        if previous_raw is not None:
            raise ValueError(
                f"duplicate raw {key_name} {values!r} on rows "
                f"{previous_raw} and {row_index}"
            )
        raw_rows[raw] = row_index

        canonical = canonical_key(values, datatypes)
        previous = canonical_rows.get(canonical)
        if previous is not None:
            previous_row, previous_raw_key = previous
            raise ValueError(
                f"normalised {key_name} collision on rows {previous_row} and "
                f"{row_index}: {previous_raw_key} and {raw}"
            )
        canonical_rows[canonical] = (row_index, raw)
        result[canonical] = row_index
    return result


__all__ = [
    "TargetKeyIndex",
    "TargetKeyLookup",
    "canonical_key",
    "normalize_key_value",
    "validate_unique_key",
]
