"""Public entry points for comparative technique research."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from kg_construction.certification.registry import PluginRegistry
from kg_construction.research.benchmark_runner import (
    TechniqueEvaluationConfig,
    TechniqueEvaluationReport,
    TechniqueEvaluationRunner,
)

PathLike: TypeAlias = str | Path
EvaluationConfigInput: TypeAlias = TechniqueEvaluationConfig | PathLike | None


def evaluate_techniques(
    input_path: PathLike,
    *,
    source_type: str | None = None,
    config: EvaluationConfigInput = None,
    techniques: str = "applicable",
    output_dir: PathLike | None = None,
    registry: PluginRegistry | None = None,
) -> TechniqueEvaluationReport:
    """Profile, execute, score, and rank construction techniques.

    This is comparative research. Its weighted ranking is not a publication
    decision and does not alter the graph returned by the construction API.
    """

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"benchmark input path does not exist: {input_path}")
    return TechniqueEvaluationRunner(
        config=_evaluation_config(config),
        registry=registry,
    ).run(
        input_path=path,
        source_type=source_type,
        techniques=techniques,
        output_dir=output_dir,
    )


def _evaluation_config(config: EvaluationConfigInput) -> TechniqueEvaluationConfig:
    if config is None:
        return TechniqueEvaluationConfig()
    if isinstance(config, TechniqueEvaluationConfig):
        return config
    return TechniqueEvaluationConfig.from_yaml(config)


__all__ = ["evaluate_techniques"]
