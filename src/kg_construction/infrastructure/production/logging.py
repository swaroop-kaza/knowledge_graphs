"""Structured JSON logging for production runs."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, TextIO

from pydantic import BaseModel, ConfigDict

from kg_construction.infrastructure.production.pii import PIIRedactionPolicy

JsonDict = dict[str, Any]

_RESERVED_LOG_RECORD_KEYS = set(
    logging.LogRecord(
        name="reserved",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="",
        args=(),
        exc_info=None,
    ).__dict__
)


class StructuredLoggingConfig(BaseModel):
    """Settings for JSON log output."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    logger_name: str = "kg_construction"
    level: str = "INFO"
    redact_pii: bool = True


class JsonLogFormatter(logging.Formatter):
    """Format stdlib log records as single-line JSON."""

    def __init__(self, *, pii_policy: PIIRedactionPolicy | None = None) -> None:
        super().__init__()
        self.pii_policy = pii_policy

    def format(self, record: logging.LogRecord) -> str:
        payload: JsonDict = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_RECORD_KEYS and key not in payload:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if self.pii_policy is not None and self.pii_policy.enabled:
            payload = self.pii_policy.redact_mapping(payload)
        return json.dumps(payload, sort_keys=True, default=str)


class StructuredLogger:
    """Small wrapper that binds run context to stdlib log events."""

    def __init__(self, logger: logging.Logger, context: Mapping[str, Any] | None = None) -> None:
        self.logger = logger
        self.context = dict(context or {})

    def bind(self, **context: Any) -> StructuredLogger:
        """Return a logger with additional bound context."""

        return StructuredLogger(self.logger, {**self.context, **context})

    def info(self, event: str, **fields: Any) -> None:
        self._log(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._log(logging.WARNING, event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._log(logging.ERROR, event, **fields)

    def exception(self, event: str, **fields: Any) -> None:
        self.logger.exception(event, extra=self._extra(event, fields))

    def _log(self, level: int, event: str, **fields: Any) -> None:
        self.logger.log(level, event, extra=self._extra(event, fields))

    def _extra(self, event: str, fields: JsonDict) -> JsonDict:
        return {"event": event, **self.context, **fields}


def configure_structured_logging(
    config: StructuredLoggingConfig | None = None,
    *,
    stream: TextIO | None = None,
    pii_policy: PIIRedactionPolicy | None = None,
) -> StructuredLogger:
    """Configure and return the package JSON logger."""

    settings = config or StructuredLoggingConfig()
    logger = logging.getLogger(settings.logger_name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(getattr(logging, settings.level.upper(), logging.INFO))
    if settings.enabled:
        handler = logging.StreamHandler(stream or sys.stdout)
        redaction_policy = pii_policy if settings.redact_pii else None
        handler.setFormatter(JsonLogFormatter(pii_policy=redaction_policy))
        logger.addHandler(handler)
    return StructuredLogger(logger)


__all__ = [
    "JsonLogFormatter",
    "StructuredLogger",
    "StructuredLoggingConfig",
    "configure_structured_logging",
]
