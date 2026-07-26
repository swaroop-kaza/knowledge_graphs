"""Comparison matrix helpers."""

from __future__ import annotations

from kg_construction.research.scoring import TechniqueScore


def build_comparison_matrix(scores: list[TechniqueScore]) -> list[dict[str, object]]:
    """Build a compact row-per-technique comparison matrix."""

    rows: list[dict[str, object]] = []
    for index, score in enumerate(scores, start=1):
        rows.append(
            {
                "rank": index,
                "technique": score.technique_name,
                "pipeline": score.pipeline_name,
                "final_score": round(score.final_score, 6),
                "quality_score": round(score.quality_score, 6),
                "runtime_score": round(score.runtime_score, 6),
                "accuracy": round(score.dimensions.get("accuracy", 0.0), 6),
                "validation": round(score.dimensions.get("validation", 0.0), 6),
                "coverage": round(score.dimensions.get("coverage", 0.0), 6),
                "cost_efficiency": round(score.dimensions.get("cost_efficiency", 0.0), 6),
                "latency": round(score.metrics.get("latency", 0.0), 6),
                "recommendation": score.recommendation,
            }
        )
    return rows


__all__ = ["build_comparison_matrix"]
