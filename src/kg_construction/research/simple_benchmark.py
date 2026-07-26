"""Benchmark runner for comparing construction techniques."""

from __future__ import annotations

import time
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.domain.models import ConstructionBatch
from kg_construction.evaluation.metrics import ConstructionMetricsEvaluator

TechniqueRunner = Callable[[], ConstructionBatch]


class BenchmarkResult(BaseModel):
    """Benchmark result for one construction technique."""

    model_config = ConfigDict(extra="forbid")

    technique: str
    metrics: dict[str, float] = Field(default_factory=dict)
    latency_seconds: float
    error: str | None = None


class BenchmarkRunner:
    """Run enabled construction methods on the same dataset."""

    name = "benchmark_runner"

    def __init__(self, evaluator: ConstructionMetricsEvaluator | None = None) -> None:
        self.evaluator = evaluator or ConstructionMetricsEvaluator()

    def compare(
        self,
        runners: dict[str, TechniqueRunner],
        *,
        expected: ConstructionBatch | None = None,
    ) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []
        for technique, runner in runners.items():
            started = time.perf_counter()
            try:
                batch = runner()
                latency = time.perf_counter() - started
                metrics = self.evaluator.evaluate(batch, expected)
                metrics["latency"] = latency
                results.append(
                    BenchmarkResult(technique=technique, metrics=metrics, latency_seconds=latency)
                )
            except Exception as exc:
                latency = time.perf_counter() - started
                results.append(
                    BenchmarkResult(
                        technique=technique,
                        metrics={},
                        latency_seconds=latency,
                        error=str(exc),
                    )
                )
        return results


__all__ = ["BenchmarkResult", "BenchmarkRunner", "TechniqueRunner"]
