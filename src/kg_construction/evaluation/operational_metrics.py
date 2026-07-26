"""Operational performance metrics."""

from __future__ import annotations

import time
import tracemalloc
from types import TracebackType

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RetryObservation(BaseModel):
    """Observed attempts for one retry-protected operation.

    Retry metrics must be derived from attempt telemetry.  In particular, the
    absence of an exception is not evidence that a retry layer executed zero
    retries.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_id: str = Field(min_length=1)
    attempt_count: int = Field(ge=1)
    retry_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _retries_are_attempts_after_the_first(self) -> RetryObservation:
        if self.retry_count != self.attempt_count - 1:
            raise ValueError(
                "retry_count must equal attempt_count - 1 for one observed operation"
            )
        return self


class RuntimeReport(BaseModel):
    """Runtime statistics for one technique execution."""

    model_config = ConfigDict(extra="forbid")

    processed_items: int = 0
    latency_seconds: float = 0.0
    throughput: float = 0.0
    memory_peak_mb: float = 0.0
    graph_write_time_seconds: float = 0.0
    retry_observations: tuple[RetryObservation, ...] = ()
    ollama_prompt_tokens: int = 0
    ollama_output_tokens: int = 0
    ollama_cost_proxy: float = 0.0
    metadata: dict[str, float | str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _retry_observation_ids_are_unique(self) -> RuntimeReport:
        operation_ids = [
            observation.operation_id for observation in self.retry_observations
        ]
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("retry observation operation_id values must be unique")
        return self


class OperationalMetricsRecorder:
    """Context manager that measures latency, throughput, and peak memory."""

    def __init__(self, *, processed_items: int = 0, compute_cost_factor: float = 0.000001) -> None:
        self.processed_items = processed_items
        self.compute_cost_factor = compute_cost_factor
        self.started = 0.0
        self.report = RuntimeReport(processed_items=processed_items)

    def __enter__(self) -> OperationalMetricsRecorder:
        tracemalloc.start()
        self.started = time.perf_counter()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        _ = current
        latency = time.perf_counter() - self.started
        throughput = self.processed_items / latency if latency > 0 else 0.0
        self.report = RuntimeReport(
            processed_items=self.processed_items,
            latency_seconds=latency,
            throughput=throughput,
            memory_peak_mb=peak / (1024 * 1024),
        )


class OperationalMetrics:
    """Convert runtime reports into scoring metrics."""

    name = "operational_metrics"

    def evaluate(self, report: RuntimeReport) -> dict[str, float]:
        latency_penalty = report.latency_seconds / (report.latency_seconds + 1.0)
        cost_efficiency = 1.0 / (1.0 + report.ollama_cost_proxy + report.latency_seconds)
        scalability = min(1.0, report.throughput / 1000.0) if report.throughput else 0.5
        metrics = {
            "throughput": report.throughput,
            "latency": report.latency_seconds,
            "memory_usage": report.memory_peak_mb,
            "graph_write_time": report.graph_write_time_seconds,
            "ollama_token_usage": float(report.ollama_prompt_tokens + report.ollama_output_tokens),
            "ollama_cost_proxy": report.ollama_cost_proxy,
            "latency_penalty": latency_penalty,
            "cost_efficiency_score": cost_efficiency,
            "scalability_score": scalability,
        }
        if report.retry_observations:
            operation_count = len(report.retry_observations)
            retry_count = sum(
                observation.retry_count
                for observation in report.retry_observations
            )
            attempt_count = sum(
                observation.attempt_count
                for observation in report.retry_observations
            )
            retried_operation_count = sum(
                observation.retry_count > 0
                for observation in report.retry_observations
            )
            metrics.update(
                {
                    "retry_rate": retried_operation_count / operation_count,
                    "retry_observation_support": float(operation_count),
                    "retry_attempt_count": float(attempt_count),
                    "retry_count": float(retry_count),
                }
            )
        return metrics


__all__ = [
    "OperationalMetrics",
    "OperationalMetricsRecorder",
    "RetryObservation",
    "RuntimeReport",
]
