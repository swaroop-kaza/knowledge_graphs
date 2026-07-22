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
            f"- Expected input hash: {report.run_metadata.expected_input_hash or 'not declared'}",
            f"- Input contract verified: {report.run_metadata.input_contract_verified}",
            "- Source inventory: "
            + (", ".join(report.run_metadata.source_inventory) or "not declared"),
            f"- Git commit: {report.run_metadata.git_commit}",
            f"- Git tree state: {report.run_metadata.git_tree_state}",
            f"- Source tree hash: {report.run_metadata.source_tree_hash}",
            f"- Gold dataset: {report.run_metadata.benchmark_id or 'not configured'}",
            f"- Gold dataset version: {report.run_metadata.benchmark_version or 'not configured'}",
            f"- Gold dataset hash: {report.run_metadata.gold_dataset_hash or 'not configured'}",
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
                "## Best-Ranked Result (Smoke Result Until Gates Pass)",
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

        lines.extend(
            [
                "",
                "## Correctness and Publication Gates",
                "",
                "A ranking is a publishable quality result only when its candidate passes every gate. "
                "Runs without compatible gold labels remain useful smoke tests but do not establish accuracy.",
                "",
                "| Technique | Gold tasks | Gate | Metric | Requirement | Actual | Result |",
                "|---|---|---|---|---:|---:|---|",
            ]
        )
        for candidate in report.candidates:
            gold_tasks = ", ".join(candidate.gold_evaluation.evaluated_tasks) or "none"
            for gate in candidate.hard_gates:
                actual = "n/a" if gate.actual is None else str(gate.actual)
                result = (
                    "NOT APPLICABLE"
                    if not gate.evaluated
                    else ("PASS" if gate.passed else "FAIL")
                )
                lines.append(
                    f"| `{candidate.technique_name}` | {gold_tasks} | {gate.gate} | "
                    f"`{gate.metric}` | {gate.operator} {gate.threshold} | {actual} | {result} |"
                )
        publication_eligible = [
            candidate.technique_name
            for candidate in report.candidates
            if candidate.publication_eligible
        ]
        lines.extend(
            [
                "",
                "- Publication-eligible candidates: "
                + (", ".join(f"`{name}`" for name in publication_eligible) or "none"),
                "- Gold tasks exposed to recommendation logic: "
                + (
                    ", ".join(
                        f"`{name}`"
                        for name in recommendations.evaluated_gold_tasks
                    )
                    or "none"
                ),
            ]
        )
        for candidate in report.candidates:
            support = ", ".join(
                f"{name}={value}"
                for name, value in sorted(candidate.gold_evaluation.support.items())
            )
            if support:
                lines.append(f"- Gold support for `{candidate.technique_name}`: {support}")

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
            if decision.confidence_interval_method == "not_estimated":
                uncertainty = "statistical score uncertainty was not estimated"
            else:
                low, high = decision.confidence_interval
                uncertainty = (
                    f"{decision.confidence_interval_method} 95% confidence interval "
                    f"{low:.3f}-{high:.3f}"
                )
            return (
                f"- {decision.technique_name} with score {decision.final_score:.3f} "
                f"({uncertainty}). {decision.reason}"
            )
        return f"- No technique selected. {decision.reason}"


__all__ = ["MarkdownReportWriter"]
