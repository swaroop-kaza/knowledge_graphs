"""Generate the committed small benchmark leaderboard."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kg_construction.application.benchmark_runner import (
    TechniqueEvaluationConfig,
    TechniqueEvaluationRunner,
)

REPORTS_DIR = Path("reports")
LEADERBOARD_MD = REPORTS_DIR / "benchmark_leaderboard.md"
LEADERBOARD_CSV = REPORTS_DIR / "benchmark_leaderboard.csv"
DEFAULT_FIXED_LATENCY_SECONDS = 0.01
DEFAULT_FIXED_MEMORY_MB = 0.25
DEFAULT_FIXED_THROUGHPUT = 1000.0


@dataclass(frozen=True, slots=True)
class BenchmarkDataset:
    """One tiny fixture used in the committed leaderboard."""

    dataset_type: str
    input_path: Path
    source_type: str | None
    techniques: str
    description: str


BENCHMARK_DATASETS = (
    BenchmarkDataset(
        dataset_type="text",
        input_path=Path("tests/fixtures/golden_benchmarks/text/input.txt"),
        source_type="text",
        techniques="regex_entity_extractor,gazetteer_entity_extractor,hybrid_rules_gazetteer_regex",
        description="Pattern-heavy operational text with email, date, incident ID, version, teams, and service names.",
    ),
    BenchmarkDataset(
        dataset_type="csv",
        input_path=Path("tests/fixtures/golden_benchmarks/csv/input.csv"),
        source_type="csv",
        techniques="direct_table_mapper,jsonpath_mapper",
        description="Small structured table with IDs, names, departments, emails, and manager references.",
    ),
    BenchmarkDataset(
        dataset_type="json",
        input_path=Path("tests/fixtures/golden_benchmarks/json/input.json"),
        source_type="json",
        techniques="jsonpath_mapper,direct_table_mapper",
        description="Nested JSON service payload with owner metadata, dependencies, and last incident details.",
    ),
    BenchmarkDataset(
        dataset_type="mixed",
        input_path=Path("tests/fixtures/golden_benchmarks/mixed"),
        source_type=None,
        techniques="direct_table_mapper,jsonpath_mapper,regex_entity_extractor,gazetteer_entity_extractor,hybrid_rules_gazetteer_regex,hybrid_structured_then_text",
        description="Directory fixture combining CSV, JSON, and text sources.",
    ),
)


def main() -> int:
    """Generate Markdown and CSV leaderboard files."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = generate_leaderboard_rows()
    _write_csv(rows, LEADERBOARD_CSV)
    _write_markdown(rows, LEADERBOARD_MD)
    print(f"benchmark-leaderboard-ok: {len(rows)} rows written")
    return 0


def generate_leaderboard_rows() -> list[dict[str, Any]]:
    """Run deterministic benchmarks and return flattened leaderboard rows."""

    rows: list[dict[str, Any]] = []
    config = _leaderboard_config()
    for dataset in BENCHMARK_DATASETS:
        report = TechniqueEvaluationRunner(config=config).run(
            input_path=dataset.input_path,
            source_type=dataset.source_type,
            techniques=dataset.techniques,
            output_dir=None,
        )
        score_by_technique = {score.technique_name: score for score in report.ranked_scores}
        candidate_by_technique = {candidate.technique_name: candidate for candidate in report.candidates}
        for rank, score in enumerate(report.ranked_scores, start=1):
            candidate = candidate_by_technique[score.technique_name]
            metrics = candidate.evaluation_report
            rows.append(
                {
                    "rank": rank,
                    "dataset_type": dataset.dataset_type,
                    "dataset_path": dataset.input_path.as_posix(),
                    "source_type": dataset.source_type or "auto",
                    "technique": score.technique_name,
                    "pipeline": score.pipeline_name,
                    "entity_coverage": _round(metrics.get("entity_coverage", 0.0)),
                    "relation_coverage": _round(metrics.get("relation_coverage", 0.0)),
                    "evidence_coverage": _round(metrics.get("evidence_coverage", 0.0)),
                    "validation_pass_rate": _round(1.0 - metrics.get("validation_failure_rate", 0.0)),
                    "latency_seconds": _round(metrics.get("latency", 0.0)),
                    "memory_mb": _round(metrics.get("memory_usage", 0.0)),
                    "throughput_items_per_second": _round(metrics.get("throughput", 0.0)),
                    "resource_usage": _resource_usage(metrics),
                    "quality_score": _round(score.quality_score),
                    "runtime_score": _round(score.runtime_score),
                    "final_score": _round(score.final_score),
                    "status": score.status,
                    "recommendation": _recommendation(rank, score.technique_name, score_by_technique),
                }
            )
    return rows


def _leaderboard_config() -> TechniqueEvaluationConfig:
    config = TechniqueEvaluationConfig()
    return config.model_copy(
        deep=True,
        update={
            "evaluation": config.evaluation.model_copy(
                update={
                    "deterministic_runtime": True,
                    "fixed_latency_seconds": DEFAULT_FIXED_LATENCY_SECONDS,
                    "fixed_memory_peak_mb": DEFAULT_FIXED_MEMORY_MB,
                    "fixed_throughput": DEFAULT_FIXED_THROUGHPUT,
                    "fixed_graph_write_time_seconds": 0.0,
                    "fixed_ollama_prompt_tokens": 0,
                    "fixed_ollama_output_tokens": 0,
                    "fixed_ollama_cost_proxy": 0.0,
                }
            )
        },
    )


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "rank",
        "dataset_type",
        "dataset_path",
        "source_type",
        "technique",
        "pipeline",
        "entity_coverage",
        "relation_coverage",
        "evidence_coverage",
        "validation_pass_rate",
        "latency_seconds",
        "memory_mb",
        "throughput_items_per_second",
        "resource_usage",
        "quality_score",
        "runtime_score",
        "final_score",
        "status",
        "recommendation",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    generated_at = datetime.now(UTC).date().isoformat()
    lines = [
        "# Small Benchmark Leaderboard",
        "",
        f"Generated: {generated_at}",
        "",
        "This committed leaderboard is intentionally small and reproducible. It uses tiny fixtures under `tests/fixtures/golden_benchmarks/`, deterministic runtime settings, and the same scoring path used by `TechniqueEvaluationRunner`. Big multi-dataset experiment outputs remain generated artifacts outside the committed repo.",
        "",
        "Runtime/resource columns are deterministic controls for comparability:",
        "",
        f"- latency: `{DEFAULT_FIXED_LATENCY_SECONDS}` seconds",
        f"- memory/resource usage: `{DEFAULT_FIXED_MEMORY_MB}` MB",
        f"- throughput: `{DEFAULT_FIXED_THROUGHPUT}` items/second",
        "",
        "Regenerate with:",
        "",
        "```bash",
        "PYTHONPATH=src python scripts/generate_benchmark_leaderboard.py",
        "```",
        "",
        "| Dataset type | Rank | Technique | Entity coverage | Relation coverage | Evidence coverage | Validation pass rate | Latency | Memory/resource | Final score |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {dataset_type} | {rank} | `{technique}` | {entity_coverage:.3f} | {relation_coverage:.3f} | {evidence_coverage:.3f} | {validation_pass_rate:.3f} | {latency_seconds:.3f}s | {memory_mb:.3f} MB | {final_score:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## How To Read This",
            "",
            "- `entity_coverage`: extracted or mapped entities divided by the profile-estimated entity count.",
            "- `relation_coverage`: extracted or mapped relations divided by the profile-estimated relation count.",
            "- `evidence_coverage`: fraction of candidates with source evidence.",
            "- `validation_pass_rate`: `1 - validation_failure_rate` from the validation metrics.",
            "- `latency` and `memory/resource`: deterministic operational controls, not hardware benchmarks.",
            "- `final_score`: weighted quality plus runtime score after penalties.",
            "",
            "## Dataset Notes",
            "",
        ]
    )
    for dataset in BENCHMARK_DATASETS:
        lines.append(f"- `{dataset.dataset_type}`: {dataset.description}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resource_usage(metrics: dict[str, float]) -> str:
    return f"{_round(metrics.get('memory_usage', 0.0)):.3f} MB"


def _recommendation(
    rank: int,
    technique_name: str,
    score_by_technique: dict[str, Any],
) -> str:
    if rank == 1:
        return "best_for_fixture"
    if technique_name.startswith("hybrid"):
        return "compare_when_mixed_signals_matter"
    best_score = next(iter(score_by_technique.values()))
    score = score_by_technique[technique_name]
    if abs(best_score.final_score - score.final_score) < 0.001:
        return "near_tie"
    return "use_when_inputs_match_strengths"


def _round(value: float) -> float:
    return round(float(value), 6)


if __name__ == "__main__":
    raise SystemExit(main())
