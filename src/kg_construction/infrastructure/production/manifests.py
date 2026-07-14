"""Run manifests for reproducible Data-to-KG executions."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.domain.models import ConstructionBatch, SourceRecord, stable_id, utc_now
from kg_construction.infrastructure.ontology.validation import ValidationReport

JsonDict = dict[str, Any]
RunStatus = Literal["running", "completed", "failed"]


def canonical_hash(payload: Any, *, length: int = 32) -> str:
    """Return a stable hash for JSON-like payloads."""

    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def git_commit() -> str:
    """Return the current git commit when available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


class RunManifest(BaseModel):
    """Durable manifest for one pipeline run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    pipeline_name: str
    status: RunStatus = "running"
    started_at: str = Field(default_factory=lambda: utc_now().isoformat())
    completed_at: str | None = None
    config_hash: str = "unknown"
    input_hash: str = "unknown"
    git_commit: str = "unknown"
    source_ids: list[str] = Field(default_factory=list)
    source_names: list[str] = Field(default_factory=list)
    counts: JsonDict = Field(default_factory=dict)
    errors: list[JsonDict] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)

    @classmethod
    def start(
        cls,
        *,
        pipeline_name: str,
        sources: list[SourceRecord],
        config_payload: Any | None = None,
        metadata: JsonDict | None = None,
    ) -> RunManifest:
        source_ids = [source.source_id for source in sources]
        input_hash = canonical_hash(
            [
                {
                    "source_id": source.source_id,
                    "source_type": source.source_type,
                    "source_name": source.source_name,
                }
                for source in sources
            ]
        )
        config_hash = canonical_hash(config_payload) if config_payload is not None else "unknown"
        run_id = stable_id("run", pipeline_name, input_hash, config_hash, utc_now().isoformat())
        return cls(
            run_id=run_id,
            pipeline_name=pipeline_name,
            config_hash=config_hash,
            input_hash=input_hash,
            git_commit=git_commit(),
            source_ids=source_ids,
            source_names=[source.source_name for source in sources],
            counts={"source_records": len(sources)},
            metadata=metadata or {},
        )

    def complete(
        self,
        *,
        batch: ConstructionBatch,
        validation_report: ValidationReport,
        metadata: JsonDict | None = None,
    ) -> RunManifest:
        """Return a completed copy with graph counts and validation counts."""

        return self.model_copy(
            update={
                "status": "completed",
                "completed_at": utc_now().isoformat(),
                "counts": {
                    **self.counts,
                    "entities": len(batch.entities),
                    "relations": len(batch.relations),
                    "facts": len(batch.facts),
                    "validation_findings": len(validation_report.findings),
                    "validation_checked_count": validation_report.checked_count,
                },
                "metadata": {**self.metadata, **(metadata or {})},
            }
        )

    def fail(self, error: BaseException, *, stage: str | None = None) -> RunManifest:
        """Return a failed copy with error details."""

        return self.model_copy(
            update={
                "status": "failed",
                "completed_at": utc_now().isoformat(),
                "errors": [
                    *self.errors,
                    {
                        "stage": stage or "unknown",
                        "error_type": type(error).__name__,
                        "message": str(error),
                    },
                ],
            }
        )


class RunManifestWriter:
    """Persist run manifests as JSON files."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, manifest: RunManifest) -> Path:
        path = self.output_dir / f"{manifest.run_id}.manifest.json"
        path.write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path


__all__ = [
    "RunManifest",
    "RunManifestWriter",
    "RunStatus",
    "canonical_hash",
    "git_commit",
]
