"""Adversarial regression tests for structured identity and category formation."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from kg_construction.application.plan_resolution import resolve_modeling_plan
from kg_construction.application.semantic_profiler import SemanticColumnProfiler
from kg_construction.application.semantic_table_mapper import SemanticTableMapper
from kg_construction.application.source_catalog import (
    CatalogTableMapper,
    DeclaredColumnMapping,
    DeclaredForeignKey,
    SourceCatalog,
    TableDeclaration,
)
from kg_construction.application.workflow import ConstructionWorkflow
from kg_construction.domain.graph_objects import PropertyGraph, PropertyNode
from kg_construction.domain.modeling import ColumnRole, ControlledVocabulary
from kg_construction.domain.models import SourceRecord
from kg_construction.infrastructure.evaluation.schema_quality import (
    SchemaQualityEvaluator,
)
from kg_construction.infrastructure.ontology.shacl import ShaclValidator


def _source(name: str, rows: list[dict[str, object]]) -> SourceRecord:
    return SourceRecord.from_rows(
        rows,
        source_type="csv",
        source_name=name,
    )


def _department_catalog() -> SourceCatalog:
    return SourceCatalog().declare(
        TableDeclaration(
            table_name="employees.csv",
            record_type="Employee",
            primary_key=["employee_id"],
            datatypes={"employee_id": "string"},
            column_mappings={
                "department": DeclaredColumnMapping(
                    role=ColumnRole.CATEGORICAL_DIMENSION,
                    materialization="node",
                    target="Department",
                    relationship_name="has_department",
                    controlled_vocabulary="DepartmentCodes",
                    concept_scheme="urn:example:scheme:department",
                )
            },
            controlled_vocabularies=[
                ControlledVocabulary(
                    name="DepartmentCodes",
                    permissible_values=["Engineering", "Sales"],
                    aliases={"eng": "Engineering"},
                    closed=True,
                )
            ],
        )
    )


def test_category_variants_share_scheme_and_canonical_code_and_retain_occurrences() -> None:
    rows = [
        {"employee_id": "001", "department": "Engineering"},
        {"employee_id": "002", "department": " ＥＮＧＩＮＥＥＲＩＮＧ "},
        {"employee_id": "003", "department": "engineering\t"},
        {"employee_id": "004", "department": "ENG"},
    ]
    source = _source("employees.csv", rows)
    mapper = CatalogTableMapper(
        _department_catalog(),
        require_declarations=True,
    )

    with patch.object(
        ShaclValidator,
        "validate_rdf_graph",
        wraps=ShaclValidator().validate_rdf_graph,
    ) as validate:
        result = ConstructionWorkflow(mappers=[mapper]).run([source])

    departments = [
        node for node in result.property_graph.nodes if "Department" in node.labels
    ]
    assert len(departments) == 1
    department = departments[0]
    assert department.properties["concept_scheme"] == "urn:example:scheme:department"
    assert department.properties["code"] == "engineering"
    assert set(department.properties["candidate_names"]) == {
        "Engineering",
        " ＥＮＧＩＮＥＥＲＩＮＧ ",
        "engineering\t",
        "ENG",
    }
    assert len(department.properties["evidence_locations"]) == 4
    assert set(department.properties["raw_values"]) == {
        "Engineering",
        " ＥＮＧＩＮＥＥＲＩＮＧ ",
        "engineering\t",
        "ENG",
    }
    assert len(department.properties["source_occurrences"]) == 4

    runtime_shapes = validate.call_args.kwargs["shacl_turtle"]
    assert (
        'sh:in ( "engineering"^^xsd:string "sales"^^xsd:string )'
        in runtime_shapes
    )
    assert result.quality_report.metric_payload()["shacl_violation_count"] == 0.0


def test_closed_vocabulary_unknown_value_fails_with_source_evidence() -> None:
    source = _source(
        "employees.csv",
        [{"employee_id": "001", "department": "Unknown"}],
    )
    mapper = CatalogTableMapper(
        _department_catalog(),
        require_declarations=True,
    )
    mapper.prepare_sources([source])

    with pytest.raises(ValueError, match=r"source_id=.*row=0.*department.*Unknown"):
        mapper.map_source(source)


def test_strict_catalog_rejects_name_inference_without_explicit_vocab_binding() -> None:
    source = _source(
        "employees.csv",
        [{"employee_id": "001", "department": "Engineering"}],
    )
    catalog = SourceCatalog().declare(
        TableDeclaration(
            table_name="employees.csv",
            record_type="Employee",
            primary_key=["employee_id"],
            controlled_vocabularies=[
                ControlledVocabulary(
                    name="Department",
                    permissible_values=["Engineering"],
                )
            ],
        )
    )

    with pytest.raises(ValueError, match="explicit.*controlled_vocabulary binding"):
        CatalogTableMapper(
            catalog,
            require_declarations=True,
        ).prepare_sources([source])


def test_runtime_shacl_in_rejects_a_non_permissible_concept_code() -> None:
    source = _source(
        "employees.csv",
        [{"employee_id": "001", "department": "Engineering"}],
    )
    base = CatalogTableMapper(
        _department_catalog(),
        require_declarations=True,
    )

    class InvalidCodeMapper:
        name = "invalid_code_mapper"

        def prepare_sources(self, sources: list[SourceRecord]) -> None:
            base.prepare_sources(sources)

        def map_source(self, item: SourceRecord):
            batch = base.map_source(item)
            entities = [
                entity.model_copy(
                    update={
                        "properties": {
                            **entity.properties,
                            "code": "not-permitted",
                        }
                    }
                )
                if entity.entity_type == "Department"
                else entity
                for entity in batch.entities
            ]
            return batch.model_copy(update={"entities": entities})

    result = ConstructionWorkflow(mappers=[InvalidCodeMapper()]).run([source])

    assert (
        result.quality_report.metric_payload()["shacl_violation_count"]
        >= 1.0
    )
    assert not result.quality_report.families["structured"].passed


def test_duplicate_category_metric_uses_normalized_scheme_and_code() -> None:
    rows = [
        {"employee_id": "001", "department": "Engineering"},
        {"employee_id": "002", "department": " engineering "},
    ]
    source = _source("employees.csv", rows)
    plan = _department_catalog().resolved_plan(source)
    graph = PropertyGraph(
        nodes=[
            PropertyNode(
                node_id="concept-a",
                labels=["Entity", "Department"],
                properties={
                    "concept_scheme": "urn:example:scheme:department",
                    "code": "Engineering",
                    "evidence_locations": [
                        {
                            "kind": "cell",
                            "row_index": 0,
                            "column": "department",
                        }
                    ],
                },
            ),
            PropertyNode(
                node_id="concept-b",
                labels=["Entity", "Department"],
                properties={
                    "concept_scheme": "URN:EXAMPLE:SCHEME:DEPARTMENT",
                    "code": " engineering ",
                    "evidence_locations": [
                        {
                            "kind": "cell",
                            "row_index": 1,
                            "column": "department",
                        }
                    ],
                },
            ),
        ]
    )

    report = SchemaQualityEvaluator().evaluate(
        graph,
        policy=plan,
        source_records=rows,
        ontology=plan.to_ontology_schema(),
        shacl_executed=True,
        shacl_violation_count=0,
    )

    assert report.categorical_duplication_rate == 0.5
    assert report.duplicate_node_rate == 0.5
    assert report.categorical_source_occurrence_coverage == 1.0
    assert not report.passed


def test_keyless_identity_is_stable_across_row_order_and_duplicate_occurrences() -> None:
    rows = [
        {"colour": "red", "shape": "round"},
        {"colour": "red", "shape": "round"},
        {"colour": "blue", "shape": "square"},
        {"colour": "blue", "shape": "square"},
    ]
    reordered = [rows[2], rows[0], rows[3], rows[1]]

    def identities(records: list[dict[str, object]]) -> tuple[set[str], set[str]]:
        source = _source("observations.csv", records)
        proposed = SemanticColumnProfiler().profile(
            records,
            table_name="observations",
            record_type="Observation",
        )
        assert proposed.identity_columns == []
        plan = resolve_modeling_plan(
            proposed,
            source_name=source.source_name,
        )
        batch = SemanticTableMapper(plan).map_source(source)
        observations = [
            entity for entity in batch.entities if entity.entity_type == "Observation"
        ]
        return (
            {
                str(entity.properties["canonical_entity_id"])
                for entity in observations
            },
            {
                str(entity.properties["source_row_id"])
                for entity in observations
            },
        )

    first_entity_ids, first_row_ids = identities(rows)
    second_entity_ids, second_row_ids = identities(reordered)
    assert len(first_entity_ids) == 4
    assert len(first_row_ids) == 4
    assert first_entity_ids == second_entity_ids
    assert first_row_ids == second_row_ids


def test_declared_primary_key_rejects_raw_duplicates() -> None:
    source = _source(
        "accounts.csv",
        [{"account_id": "001"}, {"account_id": "001"}],
    )
    mapper = CatalogTableMapper(
        SourceCatalog().declare(
            TableDeclaration(
                table_name="accounts.csv",
                record_type="Account",
                primary_key=["account_id"],
            )
        )
    )
    with pytest.raises(ValueError, match="duplicate raw primary key"):
        mapper.prepare_sources([source])


def test_leading_zero_identifier_is_lexical_unless_explicitly_numeric() -> None:
    rows = [{"account_id": "001"}, {"account_id": "1"}]
    source = _source("accounts.csv", rows)
    lexical_catalog = SourceCatalog().declare(
        TableDeclaration(
            table_name="accounts.csv",
            record_type="Account",
            primary_key=["account_id"],
        )
    )
    lexical_mapper = CatalogTableMapper(lexical_catalog)
    lexical_mapper.prepare_sources([source])
    batch = lexical_mapper.map_source(source)
    accounts = [
        entity for entity in batch.entities if entity.entity_type == "Account"
    ]
    assert len(
        {
            entity.properties["canonical_entity_id"]
            for entity in accounts
        }
    ) == 2
    assert {
        entity.properties["account_id"] for entity in accounts
    } == {"001", "1"}

    numeric_mapper = CatalogTableMapper(
        SourceCatalog().declare(
            TableDeclaration(
                table_name="accounts.csv",
                record_type="Account",
                primary_key=["account_id"],
                datatypes={"account_id": "integer"},
            )
        )
    )
    with pytest.raises(
        ValueError,
        match=r"normalised primary key .* collision",
    ):
        numeric_mapper.prepare_sources([source])


def _alternate_key_catalog() -> SourceCatalog:
    return (
        SourceCatalog()
        .declare(
            TableDeclaration(
                table_name="customers.csv",
                record_type="Customer",
                primary_key=["customer_id"],
                alternate_keys=[["country", "external_number"]],
            )
        )
        .declare(
            TableDeclaration(
                table_name="orders.csv",
                record_type="Order",
                primary_key=["order_id"],
                foreign_keys=[
                    DeclaredForeignKey(
                        columns=["country", "customer_number"],
                        target_table="customers.csv",
                        target_key=["country", "external_number"],
                        relationship_name="placed_by",
                    )
                ],
            )
        )
    )


def test_composite_alternate_key_fk_resolves_to_target_primary_node() -> None:
    customers = _source(
        "customers.csv",
        [
            {
                "customer_id": "C-001",
                "country": "US",
                "external_number": "0007",
            },
            {
                "customer_id": "C-002",
                "country": "IN",
                "external_number": "0007",
            },
        ],
    )
    orders = _source(
        "orders.csv",
        [
            {
                "order_id": "O-1",
                "country": "US",
                "customer_number": "0007",
            },
            {
                "order_id": "O-2",
                "country": "IN",
                "customer_number": "0007",
            },
        ],
    )
    graph = ConstructionWorkflow(
        mappers=[CatalogTableMapper(_alternate_key_catalog())]
    ).run([customers, orders]).property_graph

    customer_ids = {
        str(node.properties["customer_id"]): node.node_id
        for node in graph.nodes
        if "Customer" in node.labels
    }
    edges = [edge for edge in graph.edges if edge.type == "placed_by"]
    assert len(edges) == 2
    assert {edge.target_id for edge in edges} == set(customer_ids.values())
    assert all(edge.target_id in customer_ids.values() for edge in edges)


def test_duplicate_composite_alternate_target_key_is_rejected() -> None:
    customers = _source(
        "customers.csv",
        [
            {
                "customer_id": "C-001",
                "country": "US",
                "external_number": "0007",
            },
            {
                "customer_id": "C-002",
                "country": "US",
                "external_number": "0007",
            },
        ],
    )
    orders = _source(
        "orders.csv",
        [
            {
                "order_id": "O-1",
                "country": "US",
                "customer_number": "0007",
            }
        ],
    )
    mapper = CatalogTableMapper(_alternate_key_catalog())
    with pytest.raises(ValueError, match="duplicate raw alternate key"):
        mapper.prepare_sources([customers, orders])


def test_foreign_key_cannot_target_an_undeclared_candidate_key() -> None:
    catalog = SourceCatalog().declare(
        TableDeclaration(
            table_name="customers.csv",
            record_type="Customer",
            primary_key=["customer_id"],
        )
    ).declare(
        TableDeclaration(
            table_name="orders.csv",
            record_type="Order",
            primary_key=["order_id"],
            foreign_keys=[
                DeclaredForeignKey(
                    columns=["email"],
                    target_table="customers.csv",
                    target_key=["email"],
                )
            ],
        )
    )
    source = _source(
        "orders.csv",
        [{"order_id": "O-1", "email": "ada@example.test"}],
    )

    with pytest.raises(ValueError, match="not a declared primary or alternate key"):
        catalog.resolved_plan(source)
