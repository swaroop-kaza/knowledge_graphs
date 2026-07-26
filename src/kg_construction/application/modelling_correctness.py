"""Compatibility imports for evaluation-owned modelling correctness cases."""

from kg_construction.evaluation.modelling_correctness import (
    ModellingCase,
    evaluate_structured_case,
    evaluate_unstructured_case,
)

__all__ = [
    "ModellingCase",
    "evaluate_structured_case",
    "evaluate_unstructured_case",
]
