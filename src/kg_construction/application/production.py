"""Factory for production hardening services from config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kg_construction.application.config import DataToKGConfig
from kg_construction.infrastructure.production import (
    ErrorQuarantine,
    HumanReviewQueue,
    IncrementalSyncPolicy,
    PIIRedactionPolicy,
    RunManifestWriter,
    StructuredLogger,
    StructuredLoggingConfig,
    WatermarkStore,
    configure_structured_logging,
)


@dataclass(frozen=True, slots=True)
class ProductionServices:
    """Optional hardening services supplied to ConstructionWorkflow."""

    structured_logger: StructuredLogger | None = None
    manifest_writer: RunManifestWriter | None = None
    watermark_store: WatermarkStore | None = None
    incremental_sync_policy: IncrementalSyncPolicy | None = None
    quarantine: ErrorQuarantine | None = None
    human_review_queue: HumanReviewQueue | None = None
    pii_policy: PIIRedactionPolicy | None = None
    continue_on_error: bool = False

    def workflow_kwargs(self) -> dict[str, Any]:
        return {
            "structured_logger": self.structured_logger,
            "manifest_writer": self.manifest_writer,
            "watermark_store": self.watermark_store,
            "incremental_sync_policy": self.incremental_sync_policy,
            "quarantine": self.quarantine,
            "human_review_queue": self.human_review_queue,
            "pii_policy": self.pii_policy,
            "continue_on_error": self.continue_on_error,
        }


def build_production_services(config: DataToKGConfig) -> ProductionServices:
    """Create production services from validated config."""

    production = config.production
    pii_policy = PIIRedactionPolicy(**production.pii_redaction.model_dump())
    structured_logger = None
    if production.structured_logging.enabled:
        structured_logger = configure_structured_logging(
            StructuredLoggingConfig(
                enabled=production.structured_logging.enabled,
                level=production.structured_logging.level,
                redact_pii=production.structured_logging.redact_pii,
            ),
            pii_policy=pii_policy,
        )
    manifest_writer = (
        RunManifestWriter(production.run_manifest.output_dir)
        if production.run_manifest.enabled
        else None
    )
    watermark_store = (
        WatermarkStore(production.incremental_sync.watermark_path)
        if production.incremental_sync.enabled
        else None
    )
    incremental_sync_policy = (
        IncrementalSyncPolicy(watermark_field=production.incremental_sync.watermark_field)
        if production.incremental_sync.enabled
        else None
    )
    quarantine = (
        ErrorQuarantine(production.error_quarantine.path)
        if production.error_quarantine.enabled
        else None
    )
    human_review_queue = (
        HumanReviewQueue(production.human_review_queue.path)
        if production.human_review_queue.enabled
        else None
    )
    return ProductionServices(
        structured_logger=structured_logger,
        manifest_writer=manifest_writer,
        watermark_store=watermark_store,
        incremental_sync_policy=incremental_sync_policy,
        quarantine=quarantine,
        human_review_queue=human_review_queue,
        pii_policy=pii_policy,
        continue_on_error=production.continue_on_error,
    )


__all__ = ["ProductionServices", "build_production_services"]
