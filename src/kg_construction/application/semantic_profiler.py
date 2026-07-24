"""Classify each column of a table into a modelling role.

Assigns every column a :class:`~kg_construction.domain.modeling.ColumnRole`
(identifier, entity reference, categorical dimension, measure, temporal, free text,
or ignore) and emits a proposed
:class:`~kg_construction.application.modeling_policy.ModelingPolicy`. Role and
materialisation are deliberately separate: a categorical value can remain an enum
property, while a reusable business dimension can become a shared node.

Classification is a transparent, dependency-free rule set over per-column features
-- data type, distinct/row cardinality ratio, null rate, uniqueness, average text
length, and column-name heuristics -- plus foreign-key detection by value
containment. Every decision carries a confidence and a rationale for review.
"""

from __future__ import annotations

import logging
import re
from statistics import mean
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.application.modeling_policy import (
    ColumnPolicy,
    ModelingPolicy,
    PromotionThresholds,
)
from kg_construction.domain.modeling import ColumnRole
from kg_construction.domain.schema import JsonType
from kg_construction.infrastructure.mapping.direct_table_mapper import infer_column_type

JsonDict = dict[str, Any]
Materialization = Literal["identity", "edge", "node", "property", "excluded"]

logger = logging.getLogger(__name__)

# Numeric columns with at most this many distinct values are read as coded
# categories (status codes, type ids) rather than measures. Aligned with
# ydata-profiling's low-cardinality-numeric heuristic.
_NUMERIC_CATEGORICAL_MAX_DISTINCT = 10
_NEAR_UNIQUE_RATIO = 0.9

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")

_IGNORE_NAMES = frozenset(
    {"password", "passwd", "secret", "token", "api_key", "apikey", "ssn", "cvv", "credit_card"}
)
_IDENTIFIER_NAME_RE = re.compile(r"^(id|uuid|guid|pk|key)$|(_id|_uuid|_key|_pk)$", re.IGNORECASE)
_REFERENCE_NAME_RE = re.compile(r"(_id|_ref|_fk|_code|_key|_no)$", re.IGNORECASE)
_TEMPORAL_NAME_RE = re.compile(
    r"(date|time|timestamp|_at$|_on$|created|updated|modified|dob|birth|expiry|expires)",
    re.IGNORECASE,
)
_MEASURE_NAME_RE = re.compile(
    r"(amount|amt|price|cost|qty|quantity|count|total|subtotal|sum|balance|salary|wage|"
    r"revenue|profit|score|rating|rate|ratio|percent|pct|weight|height|age|duration|"
    r"distance|volume|size|_num$|number)",
    re.IGNORECASE,
)
_FREE_TEXT_NAME_RE = re.compile(
    r"(description|desc|notes?|comments?|summary|body|text|message|content|remarks?|bio|"
    r"about|reason|feedback|review|full_name|fullname|address)",
    re.IGNORECASE,
)
_CATEGORICAL_NAME_RE = re.compile(
    r"(status|state|type|category|kind|class|group|department|dept|team|role|title|level|"
    r"tier|grade|gender|sex|country|region|city|province|currency|priority|severity|label|"
    r"segment|plan|stage|phase|source|channel|method|mode|unit|color|colour|brand|make|"
    r"model|flag|active|enabled)",
    re.IGNORECASE,
)
# Cardinality is a profiling signal, not a modelling rule. Only names that usually
# denote independently reusable business dimensions are eligible for inferred node
# promotion. Scalar enums (status, priority, severity, flags, types, stages, and so
# on) stay properties unless a source declaration explicitly promotes them to a
# governed concept/entity.
_REUSABLE_DIMENSION_NAME_RE = re.compile(
    r"(department|dept|team|business_unit|organization|organisation|company|"
    r"country|region|city|province|state_province|currency|brand|manufacturer|"
    r"supplier|vendor|product|location|warehouse|store|branch|market|segment)$",
    re.IGNORECASE,
)
_REFERENCE_SUFFIX_RE = re.compile(r"(_id|_uuid|_ref|_fk|_key|_code|_no|_pk)$", re.IGNORECASE)


class ColumnFeatures(BaseModel):
    """Observed statistics for a single column."""

    model_config = ConfigDict(extra="forbid")

    column: str
    dtype: str
    row_count: int
    non_null_count: int
    distinct_count: int
    distinct_ratio: float
    null_rate: float
    is_unique: bool
    avg_text_length: float


class ColumnRoleAssignment(BaseModel):
    """The proposed role for one column, with rationale and materialisation."""

    model_config = ConfigDict(extra="forbid")

    column: str
    role: ColumnRole
    confidence: float
    materialization: Materialization
    reference_type: str | None = None
    relationship_type: str | None = None
    datatype: JsonType | None = None
    rationale: str
    features: ColumnFeatures


class ForeignKeyLink(BaseModel):
    """A detected foreign-key relationship between columns."""

    model_config = ConfigDict(extra="forbid")

    source_column: str
    target_column: str | None = None
    self_referential: bool = False


class KeyCandidate(BaseModel):
    """An inferred primary-key candidate with confidence and evidence.

    Inference proposes candidates; it never silently adopts the first unique
    column as the key. Adoption is reserved for an id-named single column or a
    jointly-unique composite when no single key exists — otherwise the candidate
    is surfaced for a declaration or reviewer to confirm.
    """

    model_config = ConfigDict(extra="forbid")

    columns: list[str]
    confidence: float
    evidence: str
    adopted: bool = False

    @property
    def composite(self) -> bool:
        return len(self.columns) > 1


class ModelingPlan(BaseModel):
    """A proposed modelling plan for one table."""

    model_config = ConfigDict(extra="forbid")

    table_name: str
    record_type: str
    identity_columns: list[str] = Field(default_factory=list)
    key_candidates: list[KeyCandidate] = Field(default_factory=list)
    foreign_keys: list[ForeignKeyLink] = Field(default_factory=list)
    assignments: list[ColumnRoleAssignment] = Field(default_factory=list)

    def roles(self) -> dict[str, ColumnRole]:
        """Return the column -> role mapping."""

        return {item.column: item.role for item in self.assignments}

    def assignment(self, column: str) -> ColumnRoleAssignment | None:
        """Return the assignment for ``column`` if present."""

        return next((item for item in self.assignments if item.column == column), None)

    def to_modeling_policy(
        self, *, thresholds: PromotionThresholds | None = None
    ) -> ModelingPolicy:
        """Lower the plan into an executable ModelingPolicy."""

        columns: list[ColumnPolicy] = []
        for item in self.assignments:
            if item.role.becomes_node:
                columns.append(
                    ColumnPolicy(
                        column=item.column,
                        role=item.role,
                        reference_type=item.reference_type,
                        relationship_type=item.relationship_type,
                        description=item.rationale,
                    )
                )
            elif item.role.becomes_property:
                columns.append(
                    ColumnPolicy(
                        column=item.column,
                        role=item.role,
                        datatype=item.datatype,
                        description=item.rationale,
                    )
                )
            else:  # identifier / ignore
                columns.append(
                    ColumnPolicy(column=item.column, role=item.role, description=item.rationale)
                )
        return ModelingPolicy(
            table_name=self.table_name,
            record_type=self.record_type,
            id_columns=list(self.identity_columns),
            columns=columns,
            thresholds=thresholds or PromotionThresholds(),
        )


def _infer_dtype(values: list[object]) -> str:
    """Infer a column type, extending the base inferrer with date/datetime."""

    non_empty = [str(value) for value in values if value not in (None, "")]
    if non_empty and all(_DATETIME_RE.match(value) for value in non_empty):
        return "datetime"
    if non_empty and all(_DATE_RE.match(value) for value in non_empty):
        return "date"
    return infer_column_type(values)


_DTYPE_TO_JSON: dict[str, JsonType] = {
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
    "string": "string",
    "date": "date",
    "datetime": "datetime",
    "unknown": "string",
}


def _singularize(token: str) -> str:
    lowered = token.lower()
    # Do not mistake non-plural endings (status, address, analysis, series) for plurals.
    if lowered.endswith(("ss", "us", "is", "sis", "series", "species")):
        return token
    if lowered.endswith("ies") and len(lowered) > 3:
        return token[:-3] + "y"
    if lowered.endswith(("ses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if lowered.endswith("s") and len(lowered) > 3:
        return token[:-1]
    return token


def _class_name(column: str) -> str:
    stripped = _REFERENCE_SUFFIX_RE.sub("", column)
    tokens = [token for token in re.split(r"[^0-9A-Za-z]+", stripped) if token]
    if not tokens:
        tokens = [token for token in re.split(r"[^0-9A-Za-z]+", column) if token] or ["Reference"]
    tokens[-1] = _singularize(tokens[-1])
    return "".join(token[:1].upper() + token[1:] for token in tokens)


def _relationship_name(column: str) -> str | None:
    stripped = _REFERENCE_SUFFIX_RE.sub("", column)
    tokens = [token for token in re.split(r"[^0-9A-Za-z]+", stripped) if token]
    if not tokens:
        return None
    return "_".join(token.lower() for token in tokens)


class SemanticColumnProfiler:
    """Classify table columns into modelling roles and propose a policy."""

    name = "semantic_column_profiler"

    def __init__(self, thresholds: PromotionThresholds | None = None) -> None:
        self.thresholds = thresholds or PromotionThresholds()

    def profile(
        self,
        records: list[JsonDict],
        *,
        table_name: str = "records",
        record_type: str = "Record",
    ) -> ModelingPlan:
        """Return a proposed modelling plan for ``records``."""

        columns = sorted({column for row in records for column in row})
        row_count = len(records)
        features = {column: self._features(column, records, row_count) for column in columns}
        identity_columns, key_candidates = self._infer_keys(columns, features, records)
        foreign_keys = self._foreign_keys(columns, records, features, identity_columns)
        fk_by_column = {link.source_column: link for link in foreign_keys}

        assignments = [
            self._classify(
                features[column],
                identity_columns=identity_columns,
                foreign_key=fk_by_column.get(column),
                record_type=record_type,
            )
            for column in columns
        ]
        logger.info(
            "Classified %d columns of %s into modelling roles (identity=%s, foreign keys=%d)",
            len(assignments),
            record_type,
            identity_columns,
            len(foreign_keys),
        )
        return ModelingPlan(
            table_name=table_name,
            record_type=record_type,
            identity_columns=identity_columns,
            key_candidates=key_candidates,
            foreign_keys=foreign_keys,
            assignments=assignments,
        )

    @staticmethod
    def _features(column: str, records: list[JsonDict], row_count: int) -> ColumnFeatures:
        values = [row.get(column) for row in records]
        non_null = [value for value in values if value not in (None, "")]
        distinct = {str(value) for value in non_null}
        text_lengths = [len(str(value)) for value in non_null]
        non_null_count = len(non_null)
        distinct_count = len(distinct)
        return ColumnFeatures(
            column=column,
            dtype=_infer_dtype(values),
            row_count=row_count,
            non_null_count=non_null_count,
            distinct_count=distinct_count,
            distinct_ratio=(distinct_count / non_null_count) if non_null_count else 0.0,
            null_rate=1.0 - (non_null_count / row_count) if row_count else 1.0,
            is_unique=non_null_count == row_count and distinct_count == row_count and row_count > 0,
            avg_text_length=mean(text_lengths) if text_lengths else 0.0,
        )

    def _infer_keys(
        self,
        columns: list[str],
        features: dict[str, ColumnFeatures],
        records: list[JsonDict],
    ) -> tuple[list[str], list[KeyCandidate]]:
        """Infer the primary key with confidence, without auto-adopting a unique column.

        An id-named unique column is adopted. A jointly-unique composite is adopted
        only when no single-column key exists. A unique but non-id-named column is
        recorded as a candidate for confirmation, never silently adopted.
        """

        unique_columns = [
            column
            for column in columns
            if features[column].is_unique and features[column].null_rate == 0.0
        ]
        id_named = [column for column in unique_columns if _IDENTIFIER_NAME_RE.search(column)]
        candidates: list[KeyCandidate] = []
        adopted: list[str] = []

        if id_named:
            adopted = [id_named[0]]
            candidates.append(
                KeyCandidate(columns=[id_named[0]], confidence=0.95, evidence="unique, id-named column", adopted=True)
            )
            for column in id_named[1:]:
                candidates.append(
                    KeyCandidate(columns=[column], confidence=0.9, evidence="unique, id-named alternate key")
                )
        else:
            for column in unique_columns:
                candidates.append(
                    KeyCandidate(
                        columns=[column],
                        confidence=0.5,
                        evidence="unique but not id-named; not auto-adopted without a declaration",
                    )
                )
            if not unique_columns:
                composite = self._composite_key(columns, features, records)
                if composite is not None:
                    adopted = composite
                    candidates.append(
                        KeyCandidate(columns=composite, confidence=0.6, evidence="jointly-unique composite key", adopted=True)
                    )
        return adopted, candidates

    @staticmethod
    def _composite_key(
        columns: list[str],
        features: dict[str, ColumnFeatures],
        records: list[JsonDict],
        *,
        max_columns: int = 8,
    ) -> list[str] | None:
        """Return the first minimal jointly-unique, non-null column pair, if any."""

        row_count = len(records)
        if row_count == 0:
            return None
        candidate_columns = [
            column
            for column in columns[:max_columns]
            if features[column].null_rate == 0.0 and not features[column].is_unique
        ]
        for left_index, left in enumerate(candidate_columns):
            for right in candidate_columns[left_index + 1 :]:
                combined = {(str(row.get(left)), str(row.get(right))) for row in records}
                if len(combined) == row_count:
                    return sorted([left, right])
        return None

    def _foreign_keys(
        self,
        columns: list[str],
        records: list[JsonDict],
        features: dict[str, ColumnFeatures],
        identity_columns: list[str],
    ) -> list[ForeignKeyLink]:
        value_sets = {
            column: {str(row.get(column)) for row in records if row.get(column) not in (None, "")}
            for column in identity_columns
        }
        links: list[ForeignKeyLink] = []
        for column in columns:
            if column in identity_columns:
                continue
            values = {str(row.get(column)) for row in records if row.get(column) not in (None, "")}
            target = next(
                (
                    identity
                    for identity, identity_values in value_sets.items()
                    if identity != column and values and values <= identity_values
                ),
                None,
            )
            if target is not None:
                links.append(
                    ForeignKeyLink(source_column=column, target_column=target, self_referential=True)
                )
            elif _REFERENCE_NAME_RE.search(column):
                links.append(ForeignKeyLink(source_column=column))
        return links

    def _classify(
        self,
        features: ColumnFeatures,
        *,
        identity_columns: list[str],
        foreign_key: ForeignKeyLink | None,
        record_type: str,
    ) -> ColumnRoleAssignment:
        column = features.column
        datatype = _DTYPE_TO_JSON.get(features.dtype, "string")

        def assign(
            role: ColumnRole,
            materialization: Materialization,
            confidence: float,
            rationale: str,
            *,
            reference_type: str | None = None,
            relationship_type: str | None = None,
            include_datatype: bool = False,
        ) -> ColumnRoleAssignment:
            return ColumnRoleAssignment(
                column=column,
                role=role,
                confidence=confidence,
                materialization=materialization,
                reference_type=reference_type,
                relationship_type=relationship_type,
                datatype=datatype if include_datatype else None,
                rationale=rationale,
                features=features,
            )

        # 1. Excluded columns.
        if column.startswith("_") or column.lower() in _IGNORE_NAMES:
            return assign(ColumnRole.IGNORE, "excluded", 0.95, "internal or prohibited column name")
        if features.non_null_count == 0:
            return assign(ColumnRole.IGNORE, "excluded", 0.9, "column is entirely null")

        # 2. Identifier (the chosen row identity only).
        if column in identity_columns:
            id_named = bool(_IDENTIFIER_NAME_RE.search(column))
            return assign(
                ColumnRole.IDENTIFIER, "identity", 0.95 if id_named else 0.8,
                "unique, id-named primary key" if id_named else "unique candidate key",
            )

        # 3. Entity reference (foreign key).
        if foreign_key is not None:
            if foreign_key.self_referential and foreign_key.target_column in identity_columns:
                reference_type = record_type
                confidence, why = 0.9, f"values are a subset of {foreign_key.target_column} (self-reference)"
            else:
                reference_type = _class_name(column)
                confidence, why = 0.7, "reference-style column name"
            return assign(
                ColumnRole.ENTITY_REFERENCE, "edge", confidence, why,
                reference_type=reference_type, relationship_type=_relationship_name(column),
            )

        # 4. Temporal.
        if features.dtype in {"date", "datetime"}:
            return assign(ColumnRole.TEMPORAL, "property", 0.9, f"{features.dtype} values", include_datatype=True)
        if _TEMPORAL_NAME_RE.search(column):
            return assign(ColumnRole.TEMPORAL, "property", 0.75, "temporal column name", include_datatype=True)

        # 5. Numeric: measure vs coded category.
        if features.dtype in {"integer", "number"}:
            if _CATEGORICAL_NAME_RE.search(column):
                return self._categorical(
                    features,
                    datatype,
                    0.85,
                    "categorical numeric-coded column name",
                    promote_to_node=bool(_REUSABLE_DIMENSION_NAME_RE.search(column)),
                )
            if _MEASURE_NAME_RE.search(column):
                return assign(ColumnRole.MEASURE, "property", 0.85, "numeric measure column name", include_datatype=True)
            if (
                features.distinct_count <= _NUMERIC_CATEGORICAL_MAX_DISTINCT
                and features.distinct_ratio < _NEAR_UNIQUE_RATIO
            ):
                return self._categorical(features, datatype, 0.6, "low-distinct numeric read as a coded category")
            return assign(ColumnRole.MEASURE, "property", 0.7, "numeric quantity", include_datatype=True)

        # 6. Boolean flag.
        if features.dtype == "boolean":
            return self._categorical(features, datatype, 0.85, "boolean flag (low-cardinality category)")

        # 7. String / unknown.
        if _FREE_TEXT_NAME_RE.search(column) or features.avg_text_length > self.thresholds.free_text_min_avg_length:
            return assign(ColumnRole.FREE_TEXT, "property", 0.8, "free-text column name or long values", include_datatype=True)
        if _CATEGORICAL_NAME_RE.search(column):
            return self._categorical(
                features,
                datatype,
                0.9,
                "categorical column name",
                promote_to_node=bool(_REUSABLE_DIMENSION_NAME_RE.search(column)),
            )
        if features.distinct_ratio >= _NEAR_UNIQUE_RATIO:
            return assign(ColumnRole.FREE_TEXT, "property", 0.6, "near-unique text (label/identifier-like value)", include_datatype=True)
        if self._is_low_cardinality(features):
            return self._categorical(features, datatype, 0.7, "low-cardinality string values")
        return assign(ColumnRole.FREE_TEXT, "property", 0.4, "unclassified string defaults to free text", include_datatype=True)

    def _categorical(
        self,
        features: ColumnFeatures,
        datatype: JsonType,
        confidence: float,
        rationale: str,
        *,
        promote_to_node: bool = False,
    ) -> ColumnRoleAssignment:
        thresholds = self.thresholds
        as_property = (
            not promote_to_node
            or not thresholds.promote_categoricals_to_nodes
            or features.distinct_count <= thresholds.keep_as_property_max_distinct
        )
        materialization: Materialization = "property" if as_property else "node"
        return ColumnRoleAssignment(
            column=features.column,
            role=ColumnRole.CATEGORICAL_DIMENSION,
            confidence=confidence,
            materialization=materialization,
            reference_type=_class_name(features.column),
            relationship_type=None,
            datatype=datatype if as_property else None,
            rationale=rationale
            + (
                " (materialise as property: no independent reusable dimension semantics)"
                if not promote_to_node
                else " (materialise as property: distinct count within supernode guard)"
                if as_property
                else " (promote to shared reference node)"
            ),
            features=features,
        )

    def _is_low_cardinality(self, features: ColumnFeatures) -> bool:
        thresholds = self.thresholds
        if features.distinct_count > thresholds.categorical_max_distinct_absolute:
            return False
        if features.non_null_count >= thresholds.min_rows_for_ratio:
            return features.distinct_ratio <= thresholds.categorical_max_distinct_ratio
        cutoff = max(
            thresholds.keep_as_property_max_distinct + 1,
            round(features.non_null_count * thresholds.categorical_max_distinct_ratio),
        )
        return features.distinct_count <= cutoff


__all__ = [
    "ColumnFeatures",
    "ColumnRoleAssignment",
    "ForeignKeyLink",
    "KeyCandidate",
    "ModelingPlan",
    "SemanticColumnProfiler",
]
