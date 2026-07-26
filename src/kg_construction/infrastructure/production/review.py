"""Human review queue for validation and merge decisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.domain.models import ConstructionBatch, stable_id, utc_now
from kg_construction.domain.validation import ValidationFinding

JsonDict = dict[str, Any]
ReviewStatus = Literal["pending_review", "approved", "rejected"]


class HumanReviewItem(BaseModel):
    """One item awaiting reviewer action."""

    model_config = ConfigDict(extra="forbid")

    review_id: str
    object_id: str
    object_type: str
    reason: str
    status: ReviewStatus = "pending_review"
    priority: int = 50
    payload: JsonDict = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: utc_now().isoformat())
    reviewed_at: str | None = None
    reviewer: str | None = None
    decision_note: str | None = None


class HumanReviewQueue:
    """Simple file-backed human review queue."""

    name = "human_review_queue"

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._items: list[HumanReviewItem] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._items = self._load()

    def route(self, findings: list[object]) -> list[dict[str, object]]:
        """Compatibility method used by registry-level validation adapters."""

        items = []
        for finding in findings:
            if isinstance(finding, ValidationFinding):
                items.append(self.enqueue_finding(finding))
            else:
                payload = {"finding": str(finding)}
                items.append(
                    self.enqueue(
                        object_id=stable_id("review_object", str(finding)),
                        object_type="unknown",
                        reason="manual_review",
                        payload=payload,
                    )
                )
        return [item.model_dump(mode="json") for item in items]

    def enqueue_finding(self, finding: ValidationFinding, *, priority: int = 50) -> HumanReviewItem:
        return self.enqueue(
            object_id=finding.object_id,
            object_type=finding.object_type,
            reason=finding.rule,
            priority=priority,
            payload={"finding": finding.model_dump(mode="json")},
        )

    def enqueue_batch_review(
        self,
        *,
        batch: ConstructionBatch,
        findings: list[ValidationFinding],
    ) -> list[HumanReviewItem]:
        _ = batch
        return [
            self.enqueue_finding(finding)
            for finding in findings
            if finding.policy == "send_to_human_review" or finding.severity == "error"
        ]

    def enqueue(
        self,
        *,
        object_id: str,
        object_type: str,
        reason: str,
        payload: JsonDict | None = None,
        priority: int = 50,
    ) -> HumanReviewItem:
        item = HumanReviewItem(
            review_id=stable_id("review", object_id, object_type, reason),
            object_id=object_id,
            object_type=object_type,
            reason=reason,
            priority=priority,
            payload=payload or {},
        )
        if not any(existing.review_id == item.review_id for existing in self._items):
            self._items.append(item)
            self._flush()
        return item

    def pending(self) -> list[HumanReviewItem]:
        return [item for item in self._items if item.status == "pending_review"]

    def decide(
        self,
        review_id: str,
        *,
        status: Literal["approved", "rejected"],
        reviewer: str,
        note: str | None = None,
    ) -> HumanReviewItem:
        for index, item in enumerate(self._items):
            if item.review_id == review_id:
                updated = item.model_copy(
                    update={
                        "status": status,
                        "reviewed_at": utc_now().isoformat(),
                        "reviewer": reviewer,
                        "decision_note": note,
                    }
                )
                self._items[index] = updated
                self._flush()
                return updated
        raise KeyError(review_id)

    def _load(self) -> list[HumanReviewItem]:
        if self.path is None or not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8") or "[]")
        return [HumanReviewItem.model_validate(item) for item in payload]

    def _flush(self) -> None:
        if self.path is None:
            return
        self.path.write_text(
            json.dumps(
                [item.model_dump(mode="json") for item in self._items],
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


__all__ = ["HumanReviewItem", "HumanReviewQueue", "ReviewStatus"]
