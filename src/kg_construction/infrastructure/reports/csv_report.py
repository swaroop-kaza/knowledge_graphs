"""CSV writer for non-authoritative technique-comparison tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class CsvReportWriter:
    """Write one stable, machine-readable row per ranked technique."""

    name = "csv_report_writer"

    FIELDNAMES = [
        "rank",
        "technique",
        "pipeline",
        "final_score",
        "quality_score",
        "runtime_score",
        "score_uncertainty_method",
        "research_gate_passed",
        "gold_tasks",
        "hard_gates_passed",
        "hard_gates_failed",
        "entity_precision",
        "entity_recall",
        "entity_f1",
        "relation_precision",
        "relation_recall",
        "relation_f1",
        "pairwise_er_precision",
        "pairwise_er_recall",
        "pairwise_er_f1",
        "false_merge_rate",
        "false_split_rate",
        "er_record_id_overlap_rate",
        "er_record_id_symmetric_coverage",
        "er_record_id_exact_match",
        "er_gold_record_support",
        "er_predicted_record_support",
        "retrieval_recall_at_k",
        "retrieval_prediction_completeness",
        "retrieval_gold_query_count",
        "retrieval_predicted_query_count",
        "retrieval_missing_query_count",
        "duplicate_edge_rate",
        "dangling_edge_rate",
        "evidence_coverage",
        "validation_failure_rate",
        "latency_seconds",
        "throughput",
        "run_id",
        "benchmark_id",
        "benchmark_version",
        "input_hash",
        "gold_dataset_hash",
        "config_hash",
        "git_commit",
        "git_tree_state",
        "source_tree_hash",
    ]

    def rows(self, report: Any) -> list[dict[str, Any]]:
        candidate_by_name = {
            candidate.technique_name: candidate
            for candidate in report.candidates
        }
        rows: list[dict[str, Any]] = []
        for rank, score in enumerate(report.ranked_scores, start=1):
            candidate = candidate_by_name[score.technique_name]
            metrics = candidate.evaluation_report
            passed = [gate.gate for gate in candidate.hard_gates if gate.passed]
            failed = [gate.gate for gate in candidate.hard_gates if not gate.passed]
            rows.append(
                {
                    "rank": rank,
                    "technique": score.technique_name,
                    "pipeline": score.pipeline_name,
                    "final_score": _rounded(score.final_score),
                    "quality_score": _rounded(score.quality_score),
                    "runtime_score": _rounded(score.runtime_score),
                    "score_uncertainty_method": score.confidence_interval_method,
                    "research_gate_passed": candidate.research_gate_passed,
                    "gold_tasks": ";".join(candidate.gold_evaluation.evaluated_tasks),
                    "hard_gates_passed": ";".join(passed),
                    "hard_gates_failed": ";".join(failed),
                    "entity_precision": _metric(metrics, "entity_precision"),
                    "entity_recall": _metric(metrics, "entity_recall"),
                    "entity_f1": _metric(metrics, "entity_f1"),
                    "relation_precision": _metric(metrics, "relation_precision"),
                    "relation_recall": _metric(metrics, "relation_recall"),
                    "relation_f1": _metric(metrics, "relation_f1"),
                    "pairwise_er_precision": _metric(metrics, "pairwise_er_precision"),
                    "pairwise_er_recall": _metric(metrics, "pairwise_er_recall"),
                    "pairwise_er_f1": _metric(metrics, "pairwise_er_f1"),
                    "false_merge_rate": _metric(metrics, "false_merge_rate"),
                    "false_split_rate": _metric(metrics, "false_split_rate"),
                    "er_record_id_overlap_rate": _metric(
                        metrics,
                        "er_record_id_overlap_rate",
                    ),
                    "er_record_id_symmetric_coverage": _metric(
                        metrics,
                        "er_record_id_symmetric_coverage",
                    ),
                    "er_record_id_exact_match": _metric(
                        metrics,
                        "er_record_id_exact_match",
                    ),
                    "er_gold_record_support": _metric(
                        metrics,
                        "er_gold_record_support",
                    ),
                    "er_predicted_record_support": _metric(
                        metrics,
                        "er_predicted_record_support",
                    ),
                    "retrieval_recall_at_k": _metric(
                        metrics,
                        "retrieval_recall_at_k",
                    ),
                    "retrieval_prediction_completeness": _metric(
                        metrics,
                        "retrieval_prediction_completeness",
                    ),
                    "retrieval_gold_query_count": _metric(
                        metrics,
                        "retrieval_gold_query_count",
                    ),
                    "retrieval_predicted_query_count": _metric(
                        metrics,
                        "retrieval_predicted_query_count",
                    ),
                    "retrieval_missing_query_count": _metric(
                        metrics,
                        "retrieval_missing_query_count",
                    ),
                    "duplicate_edge_rate": _metric(metrics, "duplicate_edge_rate"),
                    "dangling_edge_rate": _metric(metrics, "dangling_edge_rate"),
                    "evidence_coverage": _metric(metrics, "evidence_coverage"),
                    "validation_failure_rate": _metric(
                        metrics,
                        "validation_failure_rate",
                    ),
                    "latency_seconds": _metric(metrics, "latency"),
                    "throughput": _metric(metrics, "throughput"),
                    "run_id": report.run_metadata.run_id,
                    "benchmark_id": report.run_metadata.benchmark_id or "",
                    "benchmark_version": report.run_metadata.benchmark_version or "",
                    "input_hash": report.run_metadata.input_hash,
                    "gold_dataset_hash": report.run_metadata.gold_dataset_hash or "",
                    "config_hash": report.run_metadata.config_hash,
                    "git_commit": report.run_metadata.git_commit,
                    "git_tree_state": report.run_metadata.git_tree_state,
                    "source_tree_hash": report.run_metadata.source_tree_hash,
                }
            )
        return rows

    def write(self, report: Any, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            writer.writerows(self.rows(report))


def _metric(metrics: dict[str, float], name: str) -> str | float:
    if name not in metrics:
        return ""
    return _rounded(metrics[name])


def _rounded(value: float) -> float:
    return round(float(value), 6)


__all__ = ["CsvReportWriter"]
