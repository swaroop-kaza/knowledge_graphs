"""Standards-based SHACL validation through pySHACL."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from kg_construction.domain.graph_objects import RDFGraph
from kg_construction.domain.schema import GraphSchema


class ShaclValidationResult(BaseModel):
    """Result of validating an RDF graph against SHACL shapes."""

    model_config = ConfigDict(extra="forbid")

    conforms: bool
    results_text: str
    results_graph_turtle: str
    shacl_turtle: str


class ShaclValidator:
    """Validate RDF graphs against generated or supplied SHACL shapes."""

    name = "shacl_validator"
    technique = "pyshacl_rdf_validation"

    def validate_rdf_graph(
        self,
        rdf_graph: RDFGraph,
        *,
        graph_schema: GraphSchema | None = None,
        shacl_turtle: str | None = None,
        inference: str = "rdfs",
    ) -> ShaclValidationResult:
        """Validate a data graph using pySHACL."""

        if graph_schema is None and shacl_turtle is None:
            raise ValueError("Provide either graph_schema or shacl_turtle for SHACL validation.")
        shapes_turtle = shacl_turtle or (graph_schema.to_shacl_turtle() if graph_schema else "")

        from pyshacl import validate
        from rdflib import Graph

        shapes_graph = Graph()
        shapes_graph.parse(data=shapes_turtle, format="turtle")
        conforms, results_graph, results_text = validate(
            data_graph=rdf_graph.to_rdflib_graph(),
            shacl_graph=shapes_graph,
            inference=inference,
            abort_on_first=False,
            allow_infos=True,
            allow_warnings=True,
            advanced=False,
            meta_shacl=False,
        )
        return ShaclValidationResult(
            conforms=bool(conforms),
            results_text=str(results_text),
            results_graph_turtle=str(results_graph.serialize(format="turtle")),
            shacl_turtle=shapes_turtle,
        )


__all__ = ["ShaclValidationResult", "ShaclValidator"]
