"""Regression tests for evaluation ownership and legacy compatibility."""

from __future__ import annotations

from pathlib import Path

import pytest

import kg_construction.evaluation as evaluation
import kg_construction.infrastructure.evaluation as legacy_evaluation
from kg_construction.evaluation.graph_bound_suite import GraphNodeSelector
from kg_construction.evaluation.intended_use_metrics import GraphRAGCase
from kg_construction.infrastructure.evaluation.graph_bound_suite import (
    GraphNodeSelector as LegacyGraphNodeSelector,
)
from kg_construction.infrastructure.evaluation.intended_use_metrics import (
    GraphRAGCase as LegacyGraphRAGCase,
)
from kg_construction.research.simple_benchmark import BenchmarkRunner


@pytest.mark.parametrize(
    "export_name",
    [
        "AccuracyMetrics",
        "GraphNodeSelector",
        "GraphRAGCase",
        "Neo4jGraphWorkloadBackend",
        "SchemaQualityEvaluator",
    ],
)
def test_legacy_exports_are_identical_to_canonical_exports(
    export_name: str,
) -> None:
    assert getattr(legacy_evaluation, export_name) is getattr(evaluation, export_name)


def test_legacy_submodules_preserve_canonical_class_identity() -> None:
    assert LegacyGraphNodeSelector is GraphNodeSelector
    assert LegacyGraphRAGCase is GraphRAGCase
    assert GraphNodeSelector.__module__.startswith("kg_construction.evaluation")
    assert GraphRAGCase.__module__.startswith("kg_construction.evaluation")


def test_simple_benchmark_belongs_to_research_not_product_evaluation() -> None:
    assert "BenchmarkRunner" not in evaluation.__all__
    assert BenchmarkRunner.__module__ == "kg_construction.research.simple_benchmark"
    assert legacy_evaluation.BenchmarkRunner is BenchmarkRunner


def test_canonical_evaluation_initializer_has_no_legacy_or_star_imports() -> None:
    initializer = Path(evaluation.__file__).read_text(encoding="utf-8")

    assert "import *" not in initializer
    assert "infrastructure.evaluation" not in initializer
