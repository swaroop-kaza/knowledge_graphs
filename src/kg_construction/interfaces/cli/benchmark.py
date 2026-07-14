"""CLI for technique evaluation and recommendation."""

from __future__ import annotations

import argparse
from pathlib import Path

from kg_construction.application.benchmark_runner import (
    TechniqueEvaluationConfig,
    TechniqueEvaluationRunner,
)
from kg_construction.infrastructure.reports.json_report import JsonReportWriter
from kg_construction.infrastructure.reports.markdown_report import MarkdownReportWriter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate and rank KG construction techniques.")
    parser.add_argument("--input", required=True, help="Input file or directory.")
    parser.add_argument("--source-type", choices=["csv", "json", "text"], help="Optional source type override.")
    parser.add_argument("--config", default="configs/evaluation.example.yaml", help="Evaluation YAML config.")
    parser.add_argument("--techniques", default="applicable", help="'applicable', 'all', or comma-separated names.")
    parser.add_argument("--output", required=True, help="Markdown report output path.")
    parser.add_argument("--json-output", help="JSON report output path.")
    parser.add_argument("--candidate-output-dir", help="Directory for candidate graph outputs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TechniqueEvaluationConfig.from_yaml(args.config) if args.config else TechniqueEvaluationConfig()
    output_path = Path(args.output)
    json_output = Path(args.json_output) if args.json_output else output_path.with_suffix(".json")
    candidate_output_dir = (
        Path(args.candidate_output_dir)
        if args.candidate_output_dir
        else output_path.parent / "candidate_graphs"
    )

    report = TechniqueEvaluationRunner(config=config).run(
        input_path=args.input,
        source_type=args.source_type,
        techniques=args.techniques,
        output_dir=candidate_output_dir,
    )
    MarkdownReportWriter().write(report, output_path)
    JsonReportWriter().write(report, json_output)
    print(
        {
            "best_technique": report.best_technique,
            "candidate_count": len(report.candidates),
            "ranked_count": len(report.ranked_scores),
            "markdown_report": str(output_path),
            "json_report": str(json_output),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
