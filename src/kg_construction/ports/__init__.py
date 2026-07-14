"""Plugin interfaces for Data-to-KG construction."""

from __future__ import annotations

from kg_construction.ports.enrichers import GraphEnricher
from kg_construction.ports.evaluators import ConstructionEvaluator
from kg_construction.ports.extractors import EntityExtractor, RelationExtractor
from kg_construction.ports.graph_store import GraphStore, GraphWriter
from kg_construction.ports.loaders import SourceLoader
from kg_construction.ports.mappers import SourceMapper
from kg_construction.ports.preprocessors import SourcePreprocessor
from kg_construction.ports.resolvers import EntityResolver
from kg_construction.ports.validators import GraphValidator

__all__ = [
    "ConstructionEvaluator",
    "EntityExtractor",
    "EntityResolver",
    "GraphEnricher",
    "GraphStore",
    "GraphValidator",
    "GraphWriter",
    "RelationExtractor",
    "SourceLoader",
    "SourceMapper",
    "SourcePreprocessor",
]
