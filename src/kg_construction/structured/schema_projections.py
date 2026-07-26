"""Project a resolved plan into every schema artifact and prove they agree.

``project_all`` emits the six release projections from one
:class:`~kg_construction.structured.resolved_plan.ResolvedModelingPlan`.
``projection_disagreements`` then extracts class, property, relationship,
identity, datatype, and cardinality definitions back out of each *emitted
artifact* — parsing the OWL and SHACL Turtle and the Neo4j Cypher, not the
in-memory objects — and checks that every projection agrees with the plan. This
is the Stage-1 exit gate: the artifacts are a single contract seen six ways.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from kg_construction.domain.schema import GraphSchema
from kg_construction.structured.resolved_plan import (
    EvaluationExpectations,
    ResolvedModelingPlan,
    SchemaDefinitions,
)

_XSD_TO_JSON: dict[str, str] = {
    "string": "string",
    "integer": "integer",
    "decimal": "number",
    "boolean": "boolean",
    "date": "date",
    "dateTime": "datetime",
}


def _local(iri: object, namespace: str) -> str:
    text = str(iri)
    if text.startswith(namespace):
        return text[len(namespace) :]
    return text.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def _xsd_json(iri: object) -> str:
    return _XSD_TO_JSON.get(str(iri).rsplit("#", 1)[-1], "string")


class ProjectionBundle(BaseModel):
    """All six release projections derived from one resolved plan."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    property_graph_schema: GraphSchema
    jsonld_context: dict[str, object]
    owl2rl_turtle: str
    shacl_turtle: str
    neo4j_constraints: list[str]
    evaluation_expectations: EvaluationExpectations


def project_all(plan: ResolvedModelingPlan) -> ProjectionBundle:
    """Emit every release projection from ``plan``."""

    return ProjectionBundle(
        property_graph_schema=plan.to_property_graph_schema(),
        jsonld_context=plan.to_jsonld_context(),
        owl2rl_turtle=plan.to_owl2rl_turtle(),
        shacl_turtle=plan.to_shacl_shapes(),
        neo4j_constraints=plan.to_neo4j_constraints(),
        evaluation_expectations=plan.to_evaluation_expectations(),
    )


def _defs_from_graph(schema: GraphSchema) -> SchemaDefinitions:
    classes = frozenset(node.name for node in schema.node_types)
    properties = frozenset(
        (node.name, prop.name, prop.value_type)
        for node in schema.node_types
        for prop in node.properties
    )
    relationships = frozenset(
        (rel.name, rel.source_types[0], rel.target_types[0])
        for rel in schema.relationship_types
        if rel.source_types and rel.target_types
    )
    identity = frozenset(
        (node.name, tuple(node.primary_keys)) for node in schema.node_types if node.primary_keys
    )
    return SchemaDefinitions(
        classes=classes, properties=properties, relationships=relationships, identity=identity
    )


def _defs_from_owl(turtle: str, namespace: str) -> SchemaDefinitions:
    from rdflib import OWL, RDF, RDFS, Graph

    graph = Graph()
    graph.parse(data=turtle, format="turtle")
    classes = frozenset(_local(s, namespace) for s in graph.subjects(RDF.type, OWL.Class))
    properties: set[tuple[str, str, str]] = set()
    relationships: set[tuple[str, str, str]] = set()
    for prop in graph.subjects(RDF.type, OWL.DatatypeProperty):
        domain = graph.value(prop, RDFS.domain)
        range_ = graph.value(prop, RDFS.range)
        if domain is not None and range_ is not None:
            properties.add((_local(domain, namespace), _local(prop, namespace), _xsd_json(range_)))
    for rel in graph.subjects(RDF.type, OWL.ObjectProperty):
        domain = graph.value(rel, RDFS.domain)
        range_ = graph.value(rel, RDFS.range)
        if domain is not None and range_ is not None:
            relationships.add((_local(rel, namespace), _local(domain, namespace), _local(range_, namespace)))
    return SchemaDefinitions(
        classes=classes, properties=frozenset(properties), relationships=frozenset(relationships)
    )


def _defs_from_shacl(turtle: str, namespace: str) -> SchemaDefinitions:
    from rdflib import RDF, Graph, Namespace, URIRef

    sh = Namespace("http://www.w3.org/ns/shacl#")
    graph = Graph()
    graph.parse(data=turtle, format="turtle")
    classes: set[str] = set()
    properties: set[tuple[str, str, str]] = set()
    relationships: set[tuple[str, str, str]] = set()
    cardinality: set[tuple[str, int, int]] = set()
    for shape in graph.subjects(RDF.type, sh.NodeShape):
        target = graph.value(shape, sh.targetClass)
        if target is None:
            continue
        class_name = _local(target, namespace)
        classes.add(class_name)
        for prop_shape in graph.objects(shape, sh.property):
            path = graph.value(prop_shape, sh.path)
            if path is None:
                continue
            name = _local(path, namespace)
            datatype = graph.value(prop_shape, sh.datatype)
            object_class = graph.value(prop_shape, URIRef(str(sh) + "class"))
            if datatype is not None:
                properties.add((class_name, name, _xsd_json(datatype)))
            elif object_class is not None:
                relationships.add((name, class_name, _local(object_class, namespace)))
            min_count = graph.value(prop_shape, sh.minCount)
            max_count = graph.value(prop_shape, sh.maxCount)
            cardinality.add(
                (
                    name,
                    int(str(min_count)) if min_count is not None else 0,
                    int(str(max_count)) if max_count is not None else -1,
                )
            )
    return SchemaDefinitions(
        classes=frozenset(classes),
        properties=frozenset(properties),
        relationships=frozenset(relationships),
        cardinality=frozenset(cardinality),
    )


_CONSTRAINT_RE = re.compile(
    r"FOR \(n:(?P<label>\w+)\) REQUIRE (?P<body>.+?) IS (?P<kind>UNIQUE|NODE KEY);"
)


def _defs_from_neo4j(constraints: list[str]) -> SchemaDefinitions:
    identity: set[tuple[str, tuple[str, ...]]] = set()
    for statement in constraints:
        match = _CONSTRAINT_RE.search(statement)
        if not match:
            continue
        keys = tuple(part.strip().removeprefix("n.") for part in match["body"].strip("()").split(","))
        identity.add((match["label"], tuple(key.strip() for key in keys)))
    return SchemaDefinitions(identity=frozenset(identity))


def _defs_from_eval(expectations: EvaluationExpectations) -> SchemaDefinitions:
    classes = frozenset(expectations.classes)
    properties = frozenset(
        (key.split(".", 1)[0], key.split(".", 1)[1], datatype)
        for key, datatype in expectations.property_datatypes.items()
    )
    relationships = frozenset(
        (rel["name"], rel["domain"], rel["range"]) for rel in expectations.relationships
    )
    identity = frozenset((cls, tuple(keys)) for cls, keys in expectations.identity.items())
    return SchemaDefinitions(
        classes=classes, properties=properties, relationships=relationships, identity=identity
    )


def _jsonld_disagreements(plan: ResolvedModelingPlan) -> list[str]:
    context = plan.to_jsonld_context()["@context"]
    assert isinstance(context, dict)
    classes = {item.name for item in plan.classes}
    ld_classes = {
        term
        for term, value in context.items()
        if isinstance(value, dict) and "@type" not in value and term in classes
    }
    issues: list[str] = []
    if ld_classes != classes:
        issues.append(f"jsonld classes {sorted(ld_classes)} != plan {sorted(classes)}")
    for prop in plan.properties:
        entry = context.get(prop.name)
        expected = "http://www.w3.org/2001/XMLSchema#" + {
            "number": "decimal",
            "datetime": "dateTime",
        }.get(prop.datatype, prop.datatype)
        if not isinstance(entry, dict) or entry.get("@type") != expected:
            issues.append(f"jsonld property {prop.name} @type mismatch")
    for rel in plan.relationships:
        entry = context.get(rel.name)
        if not isinstance(entry, dict) or entry.get("@type") != "@id":
            issues.append(f"jsonld relationship {rel.name} is not an @id term")
    return issues


def projection_disagreements(plan: ResolvedModelingPlan) -> list[str]:
    """Return every way the six projections disagree with the plan (empty = agree)."""

    canonical = plan.definitions()
    bundle = project_all(plan)
    extracted = {
        "property_graph_schema": _defs_from_graph(bundle.property_graph_schema),
        "owl2rl": _defs_from_owl(bundle.owl2rl_turtle, plan.namespace),
        "shacl": _defs_from_shacl(bundle.shacl_turtle, plan.namespace),
        "neo4j_constraints": _defs_from_neo4j(bundle.neo4j_constraints),
        "evaluation_expectations": _defs_from_eval(bundle.evaluation_expectations),
    }
    issues: list[str] = []
    for name, definitions in extracted.items():
        expected = canonical.restricted_to(definitions)
        if definitions != expected:
            for facet in ("classes", "properties", "relationships", "identity", "cardinality"):
                got = getattr(definitions, facet)
                want = getattr(expected, facet)
                if got != want:
                    issues.append(f"{name}.{facet}: missing={sorted(want - got)} extra={sorted(got - want)}")
    issues.extend(_jsonld_disagreements(plan))
    return issues


def assert_projections_agree(plan: ResolvedModelingPlan) -> None:
    """Raise if any projection disagrees with the plan on the six facets."""

    issues = projection_disagreements(plan)
    if issues:
        raise ValueError("schema projections disagree:\n  " + "\n  ".join(issues))


__all__ = [
    "ProjectionBundle",
    "assert_projections_agree",
    "project_all",
    "projection_disagreements",
]
