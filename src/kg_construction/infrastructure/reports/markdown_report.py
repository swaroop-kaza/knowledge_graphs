"""Markdown report renderer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from kg_construction.infrastructure.reports.comparison_matrix import build_comparison_matrix


class MarkdownReportWriter:
    """Render a human-readable KG construction technique evaluation report."""

    name = "markdown_report_writer"

    def render(self, report: Any) -> str:
        profile = report.dataset_profile
        recommendations = report.recommendations
        rows = build_comparison_matrix(report.ranked_scores)
        lines = [
            "# KG Construction Technique Evaluation Report",
            "",
            "## Benchmark Metadata",
            f"- Run ID: {report.run_metadata.run_id}",
            f"- Timestamp: {report.run_metadata.timestamp}",
            f"- Config hash: {report.run_metadata.config_hash}",
            f"- Input hash: {report.run_metadata.input_hash}",
            f"- Git commit: {report.run_metadata.git_commit}",
            f"- Deterministic runtime: {report.run_metadata.deterministic_runtime}",
            f"- Tie-breaking policy: {', '.join(report.run_metadata.tie_breaking_policy)}",
            "",
            "## Dataset Profile",
            f"- Data type: {profile.data_type}",
            f"- File format: {profile.file_format}",
            f"- Structure score: {profile.structuredness_score:.3f}",
            f"- Schema quality: {profile.schema_quality_score:.3f}",
            f"- Text complexity: {profile.text_complexity_score:.3f}",
            f"- Entity density: {profile.entity_density_score:.3f}",
            f"- Relation density: {profile.relation_density_score:.3f}",
            f"- Nestedness: {profile.nestedness_score:.3f}",
            "",
            "## Ranked Techniques",
            "",
            "| Rank | Technique | Final Score | Quality | Runtime | Accuracy | Validation | Coverage | Cost | Latency | Recommendation |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in rows:
            rank = cast(int, row["rank"])
            final_score = cast(float, row["final_score"])
            quality_score = cast(float, row["quality_score"])
            runtime_score = cast(float, row["runtime_score"])
            accuracy = cast(float, row["accuracy"])
            validation = cast(float, row["validation"])
            coverage = cast(float, row["coverage"])
            cost_efficiency = cast(float, row["cost_efficiency"])
            latency = cast(float, row["latency"])
            lines.append(
                f"| {rank} | {row['technique']} | {final_score:.3f} | {quality_score:.3f} | "
                f"{runtime_score:.3f} | {accuracy:.3f} | "
                f"{validation:.3f} | {coverage:.3f} | {cost_efficiency:.3f} | "
                f"{latency:.3f} | {row['recommendation']} |"
            )

        lines.extend(
            [
                "",
                "## Best Technique",
                self._decision_line(recommendations.best_single_technique),
                "",
                "## Best Hybrid Pipeline",
                self._decision_line(recommendations.best_hybrid_technique),
                "",
                "## Production Recommendation",
                self._decision_line(recommendations.production_safe_technique),
            ]
        )
        if recommendations.fallback_recommendation:
            lines.append(f"- Fallback: {recommendations.fallback_recommendation}")

        failed = [candidate for candidate in report.candidates if candidate.error]
        lines.extend(["", "## Failed or Weak Techniques"])
        if failed:
            for candidate in failed:
                lines.append(f"- {candidate.technique_name}: {candidate.error}")
        else:
            lines.append("- No executable candidate failed.")

        lines.extend(
            [
                "",
                "## Validation Summary",
                f"- Candidate count: {len(report.candidates)}",
                f"- Applicability results: {len(report.applicability_results)}",
                f"- Best validation score: {max((score.dimensions.get('validation', 0.0) for score in report.ranked_scores), default=0.0):.3f}",
                f"- Best evidence coverage: {max((score.metrics.get('evidence_coverage', 0.0) for score in report.ranked_scores), default=0.0):.3f}",
                "",
                "## Mathematical Scoring",
                "FinalScore(t) = quality_score(t) + runtime_score(t). Quality contains accuracy, validation, coverage, entity resolution, retrieval, maintainability, unsupported-extraction, schema, and human-review signals. Runtime contains scalability, cost efficiency, and latency signals.",
                "",
                "## Limitations",
            ]
        )
        if recommendations.limitations:
            lines.extend(f"- {limitation}" for limitation in recommendations.limitations)
        else:
            lines.append("- No major limitations were reported.")
        return "\n".join(lines) + "\n"

    def write(self, report: Any, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.render(report), encoding="utf-8")

    @staticmethod
    def _decision_line(decision: Any) -> str:
        if decision.technique_name:
            low, high = decision.confidence_interval
            return (
                f"- {decision.technique_name} with score {decision.final_score:.3f} "
                f"(confidence interval {low:.3f}-{high:.3f}). {decision.reason}"
            )
        return f"- No technique selected. {decision.reason}"


__all__ = ["MarkdownReportWriter"]
