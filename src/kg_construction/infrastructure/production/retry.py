"""Batch retry policy for idempotent production writes."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RetryPolicyConfig:
    """Configuration for bounded exponential retry."""

    max_attempts: int = 3
    initial_delay_seconds: float = 0.1
    backoff_multiplier: float = 2.0
    max_delay_seconds: float = 5.0
    sleep: bool = True
    retryable_error_names: tuple[str, ...] = (
        "ConnectionError",
        "OSError",
        "ServiceUnavailable",
        "SessionExpired",
        "TimeoutError",
        "TransientError",
    )
    retryable_message_fragments: tuple[str, ...] = (
        "connection",
        "deadlock",
        "lock",
        "retry",
        "temporarily",
        "timeout",
        "transient",
        "unavailable",
    )

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")
        if self.initial_delay_seconds < 0:
            raise ValueError("initial_delay_seconds cannot be negative.")
        if self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be at least 1.")
        if self.max_delay_seconds < 0:
            raise ValueError("max_delay_seconds cannot be negative.")


@dataclass(frozen=True, slots=True)
class RetryOutcome(Generic[T]):
    """Result metadata for a successful retry-protected operation."""

    value: T
    attempts: int
    retry_count: int


class RetryExhaustedError(RuntimeError):
    """Raised when retry attempts are exhausted."""

    def __init__(self, operation_name: str, attempts: int, last_error: BaseException) -> None:
        self.operation_name = operation_name
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"{operation_name} failed after {attempts} attempt(s): {type(last_error).__name__}: {last_error}"
        )


class BatchRetryPolicy:
    """Retry idempotent batch operations with bounded exponential backoff."""

    def __init__(self, config: RetryPolicyConfig | None = None) -> None:
        self.config = config or RetryPolicyConfig()

    def run(self, operation: Callable[[], T], *, operation_name: str) -> RetryOutcome[T]:
        last_error: BaseException | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                return RetryOutcome(
                    value=operation(),
                    attempts=attempt,
                    retry_count=attempt - 1,
                )
            except BaseException as exc:
                last_error = exc
                if attempt >= self.config.max_attempts or not self.should_retry(exc):
                    raise RetryExhaustedError(operation_name, attempt, exc) from exc
                self._sleep(attempt)
        raise RetryExhaustedError(operation_name, self.config.max_attempts, last_error or RuntimeError("unknown"))

    def should_retry(self, error: BaseException) -> bool:
        error_name = type(error).__name__
        if error_name in self.config.retryable_error_names:
            return True
        message = str(error).casefold()
        return any(fragment in message for fragment in self.config.retryable_message_fragments)

    def _sleep(self, attempt: int) -> None:
        if not self.config.sleep:
            return
        delay = min(
            self.config.initial_delay_seconds * (self.config.backoff_multiplier ** (attempt - 1)),
            self.config.max_delay_seconds,
        )
        if delay > 0:
            time.sleep(delay)


__all__ = [
    "BatchRetryPolicy",
    "RetryExhaustedError",
    "RetryOutcome",
    "RetryPolicyConfig",
]
