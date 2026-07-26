"""Compatibility facade for the canonical evaluation and research packages.

Product evaluation implementations moved to ``kg_construction.evaluation``.
The former simple technique benchmark moved to
``kg_construction.research.simple_benchmark``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from kg_construction.evaluation import __all__ as _EVALUATION_EXPORTS

_RESEARCH_EXPORTS = {
    "BenchmarkResult",
    "BenchmarkRunner",
    "TechniqueRunner",
}


def __getattr__(name: str) -> Any:
    """Resolve a former infrastructure export from its canonical owner."""

    if name in _RESEARCH_EXPORTS:
        module = import_module("kg_construction.research.simple_benchmark")
    elif name in _EVALUATION_EXPORTS:
        module = import_module("kg_construction.evaluation")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = sorted([*_EVALUATION_EXPORTS, *_RESEARCH_EXPORTS])
