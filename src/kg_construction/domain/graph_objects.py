"""Portable property graph and RDF graph objects."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.domain.models import stable_id

JsonDict = dict[str, Any]
RDFObjectType = Literal["iri", "literal"]


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


class GraphObject(BaseModel):
    """Base model for serializable graph objects."""

    model_config = ConfigDict(extra="forbid")


class PropertyNode(GraphObject):
    """A labeled property graph node."""

    node_id: str
    labels: list[str]
    properties: JsonDict = Field(default_factory=dict)


class PropertyEdge(GraphObject):
    """A labeled property graph relationship."""

    edge_id: str
    source_id: str
    target_id: str
    type: str
    properties: JsonDict = Field(default_factory=dict)


class PropertyGraph(GraphObject):
    """Portable property graph that can be exported to graph stores."""

    nodes: list[PropertyNode] = Field(default_factory=list)
    edges: list[PropertyEdge] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return self.model_dump(mode="json")

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=_json_default)


class RDFTriple(GraphObject):
    """RDF triple."""

    subject: str
    predicate: str
    object: str
    object_type: RDFObjectType = "literal"
    datatype: str | None = None

    @property
    def triple_id(self) -> str:
        return stable_id("triple", self.subject, self.predicate, self.object, self.object_type, self.datatype or "")


class RDFQuad(GraphObject):
    """RDF quad with named graph support."""

    subject: str
    predicate: str
    object: str
    graph: str
    object_type: RDFObjectType = "literal"
    datatype: str | None = None

    @property
    def quad_id(self) -> str:
        return stable_id("quad", self.subject, self.predicate, self.object, self.graph, self.object_type, self.datatype or "")


class RDFGraph(GraphObject):
    """RDF graph backed by RDFLib serialization and named-graph quads."""

    triples: list[RDFTriple] = Field(default_factory=list)
    quads: list[RDFQuad] = Field(default_factory=list)
    prefixes: dict[str, str] = Field(
        default_factory=lambda: {
            "kg": "urn:kg:schema:",
            "res": "urn:kg:",
            "prov": "http://www.w3.org/ns/prov#",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
        }
    )
    metadata: JsonDict = Field(default_factory=dict)

    def to_rdflib_graph(self):
        """Return an RDFLib Graph containing default graph triples."""

        from rdflib import Graph

        graph = Graph()
        self._bind_prefixes(graph)
        for triple in self.triples:
            graph.add(self._rdflib_triple(triple))
        return graph

    def to_rdflib_dataset(self):
        """Return an RDFLib Dataset containing triples plus named graph quads."""

        from rdflib import Dataset, URIRef

        dataset = Dataset()
        self._bind_prefixes(dataset)
        default_graph = dataset.default_graph
        for triple in self.triples:
            default_graph.add(self._rdflib_triple(triple))
        for quad in self.quads:
            named_graph = dataset.graph(URIRef(quad.graph))
            named_graph.add(self._rdflib_triple(quad))
        return dataset

    def to_turtle_lines(self) -> list[str]:
        """Return standards-compliant Turtle serialization lines."""

        turtle = self.serialize_turtle()
        return turtle.splitlines()

    def to_nquads_lines(self) -> list[str]:
        """Return standards-compliant N-Quads serialization lines."""

        nquads = self.serialize_nquads()
        return nquads.splitlines()

    def serialize_turtle(self) -> str:
        """Serialize default graph triples as Turtle through RDFLib."""

        return str(self.to_rdflib_graph().serialize(format="turtle"))

    def serialize_nquads(self) -> str:
        """Serialize named graph quads as N-Quads through RDFLib."""

        dataset = self.to_rdflib_dataset()
        return str(dataset.serialize(format="nquads"))

    def to_dict(self) -> JsonDict:
        return self.model_dump(mode="json")

    def _bind_prefixes(self, graph: Any) -> None:
        for prefix, iri in sorted(self.prefixes.items()):
            graph.bind(prefix, iri)

    @staticmethod
    def _rdflib_triple(triple: RDFTriple | RDFQuad):
        from rdflib import Literal, URIRef
        from rdflib.term import Node

        subject = URIRef(triple.subject)
        predicate = URIRef(triple.predicate)
        rdf_object: Node
        if triple.object_type == "iri":
            rdf_object = URIRef(triple.object)
        else:
            datatype = URIRef(triple.datatype) if triple.datatype else None
            rdf_object = Literal(triple.object, datatype=datatype)
        return subject, predicate, rdf_object


__all__ = [
    "PropertyEdge",
    "PropertyGraph",
    "PropertyNode",
    "RDFGraph",
    "RDFObjectType",
    "RDFQuad",
    "RDFTriple",
]
