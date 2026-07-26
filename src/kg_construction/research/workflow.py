"""Explicitly non-publishable construction helpers for comparative research.

This module is intentionally absent from the package-level public exports.
"""

from __future__ import annotations

from typing import Any, cast

from kg_construction.certification.publication_contract import TrustedPublicationBinding
from kg_construction.construction.api import _workflow_for_source_type
from kg_construction.construction.config import DataToKGConfig
from kg_construction.construction.workflow import DraftConstructionWorkflow
from kg_construction.evaluation import VersionedGraphEvaluationSuite
from kg_construction.service_runtime import LockedServiceRuntime
from kg_construction.structured.source_catalog import SourceCatalog


def draft_workflow_for_source_type(
    source_type: str | None,
    config: DataToKGConfig | None = None,
    *,
    source_catalog: SourceCatalog | None = None,
    graph_evaluation_suite: VersionedGraphEvaluationSuite | None = None,
    publication_contract: TrustedPublicationBinding | None = None,
    service_runtime: LockedServiceRuntime | None = None,
    production_kwargs: dict[str, Any] | None = None,
) -> DraftConstructionWorkflow:
    """Create an explicitly non-publishable workflow for tests and research."""

    return cast(
        DraftConstructionWorkflow,
        _workflow_for_source_type(
            source_type,
            config,
            source_catalog=source_catalog,
            graph_evaluation_suite=graph_evaluation_suite,
            publication_contract=publication_contract,
            service_runtime=service_runtime,
            production_kwargs=production_kwargs,
            workflow_class=DraftConstructionWorkflow,
        ),
    )


__all__ = ["draft_workflow_for_source_type"]
