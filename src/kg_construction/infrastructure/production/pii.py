"""PII redaction policy for logs, manifests, and source payloads."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict

from kg_construction.domain.models import SourceRecord

JsonDict = dict[str, Any]


class PIIRedactionPolicy(BaseModel):
    """Deterministic redaction rules for common sensitive values."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    replacement: str = "[REDACTED]"
    redact_emails: bool = True
    redact_phone_numbers: bool = True
    redact_ssn: bool = True
    redact_ip_addresses: bool = True
    redact_api_keys: bool = True

    def redact_text(self, text: str) -> str:
        """Redact configured sensitive patterns from text."""

        if not self.enabled:
            return text

        redacted = text
        if self.redact_emails:
            redacted = re.sub(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
                self.replacement,
                redacted,
            )
        if self.redact_phone_numbers:
            redacted = re.sub(
                r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\d)",
                self.replacement,
                redacted,
            )
        if self.redact_ssn:
            redacted = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", self.replacement, redacted)
        if self.redact_ip_addresses:
            redacted = re.sub(
                r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b",
                self.replacement,
                redacted,
            )
        if self.redact_api_keys:
            redacted = re.sub(
                r"(?i)\b(api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[^'\"\s,;]+",
                lambda match: f"{match.group(1)}={self.replacement}",
                redacted,
            )
        return redacted

    def redact_value(self, value: Any) -> Any:
        """Recursively redact strings inside JSON-like values."""

        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, dict):
            return {key: self.redact_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.redact_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.redact_value(item) for item in value)
        return value

    def redact_mapping(self, payload: JsonDict) -> JsonDict:
        """Return a redacted copy of a mapping."""

        return {key: self.redact_value(value) for key, value in payload.items()}

    def redact_source_record(self, source: SourceRecord) -> SourceRecord:
        """Return a source record with content, rows, and metadata redacted."""

        return source.model_copy(
            update={
                "content": self.redact_text(source.content) if source.content is not None else None,
                "records": [self.redact_mapping(row) for row in source.records],
                "metadata": self.redact_mapping(source.metadata),
            }
        )


__all__ = ["PIIRedactionPolicy"]
