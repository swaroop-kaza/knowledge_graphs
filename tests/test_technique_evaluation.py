from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kg_construction.application.benchmark_runner import (
    TechniqueEvaluationConfig,
    TechniqueEvaluationRunner,
)
from kg_construction.application.dataset_profiler import DatasetProfiler
from kg_construction.application.registry import default_registry
from kg_construction.application.scoring import ScoringEngine
from kg_construction.application.technique_selector import TechniqueSelector
from kg_construction.infrastructure.evaluation.statistical_confidence import wilson_score_interval
from kg_construction.infrastructure.loaders.csv_loader import CsvLoader
from kg_construction.infrastructure.reports.markdown_report import MarkdownReportWriter
from kg_construction.interfaces.cli.benchmark import main as benchmark_main

GOLDEN_FIXTURES = Path("tests/fixtures/golden_benchmarks")


class TechniqueEvaluationTests(unittest.TestCase):
    def test_profiler_detects_structured_keys_and_scores(self) -> None:
        sources = CsvLoader().load("examples/data_to_kg/sample.csv")
        profile = DatasetProfiler().profile_sources(sources, input_path="examples/data_to_kg/sample.csv")

        self.assertEqual(profile.data_type, "structured")
        self.assertIn("id", profile.primary_keys)
        self.assertIn(("manager_id", "id"), profile.foreign_keys)
        self.assertGreater(profile.structuredness_score, 0.8)
        self.assertGreater(profile.schema_quality_score, 0.5)

    def test_applicability_prefers_direct_mapping_for_csv(self) -> None:
        sources = CsvLoader().load("examples/data_to_kg/sample.csv")
        profile = DatasetProfiler().profile_sources(sources, input_path="examples/data_to_kg/sample.csv")
        selector = TechniqueSelector(registry=default_registry(), threshold=0.0)

        direct = selector.score(default_registry().get("direct_table_mapper"), profile)
        regex = selector.score(default_registry().get("regex_entity_extractor"), profile)

        self.assertGreater(direct.score, regex.score)
        self.assertGreaterEqual(direct.components["schema_match"], 0.5)

    def test_weighted_scoring_penalizes_unsupported_extraction_and_latency(self) -> None:
        engine = ScoringEngine()
        good = engine.score(
            technique_name="good",
            pipeline_name="unit",
            metrics={
                "entity_f1": 0.9,
                "relation_f1": 0.8,
                "schema_validity_score": 1.0,
                "evidence_coverage": 1.0,
                "provenance_coverage": 1.0,
                "entity_coverage": 0.8,
                "relation_coverage": 0.7,
                "source_coverage": 1.0,
                "attribute_coverage": 0.9,
                "cost_efficiency_score": 0.9,
                "scalability_score": 0.8,
                "unsupported_extraction_rate": 0.0,
                "latency_penalty": 0.05,
            },
        )
        weak = engine.score(
            technique_name="weak",
            pipeline_name="unit",
            metrics={
                **good.metrics,
                "unsupported_extraction_rate": 0.5,
                "latency_penalty": 0.8,
            },
        )

        self.assertGreater(good.final_score, weak.final_score)
        self.assertGreater(good.quality_score, weak.quality_score)
        self.assertGreater(good.runtime_score, weak.runtime_score)

    def test_wilson_interval_is_bounded(self) -> None:
        low, high = wilson_score_interval(9, 10)

        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)
        self.assertLess(low, high)

    def test_runner_compares_three_or_more_text_techniques(self) -> None:
        report = TechniqueEvaluationRunner(config=TechniqueEvaluationConfig()).run(
            input_path="examples/data_to_kg/sample.txt",
            source_type="text",
            techniques="applicable",
        )

        self.assertGreaterEqual(len(report.candidates), 3)
        self.assertGreaterEqual(len(report.ranked_scores), 3)
        self.assertIsNotNone(report.best_technique)
        self.assertEqual(
            report.ranked_scores,
            sorted(report.ranked_scores, key=lambda item: item.final_score, reverse=True),
        )
        self.assertIn(
            report.best_technique,
            {candidate.technique_name for candidate in report.candidates},
        )

    def test_deterministic_benchmark_mode_repeats_rankings_and_scores(self) -> None:
        runner = TechniqueEvaluationRunner(config=TechniqueEvaluationConfig())
        first = runner.run(
            input_path=GOLDEN_FIXTURES / "text" / "input.txt",
            source_type="text",
            techniques="applicable",
        )
        second = runner.run(
            input_path=GOLDEN_FIXTURES / "text" / "input.txt",
            source_type="text",
            techniques="applicable",
        )

        first_scores = [
            (score.technique_name, score.final_score, score.quality_score, score.runtime_score)
            for score in first.ranked_scores
        ]
        second_scores = [
            (score.technique_name, score.final_score, score.quality_score, score.runtime_score)
            for score in second.ranked_scores
        ]

        self.assertEqual(first_scores, second_scores)
        self.assertEqual(first.run_metadata.input_hash, second.run_metadata.input_hash)
        self.assertEqual(first.run_metadata.config_hash, second.run_metadata.config_hash)
        self.assertTrue(first.run_metadata.deterministic_runtime)
        self.assertEqual(
            first.run_metadata.tie_breaking_policy,
            [
                "final_score_desc",
                "production_thresholds_pass_desc",
                "latency_asc",
                "maintainability_desc",
                "technique_name_asc",
            ],
        )
        self.assertTrue(all(score.metrics["latency"] == 0.0 for score in first.ranked_scores))

    def test_golden_benchmark_rankings_cover_text_csv_json_and_mixed_input(self) -> None:
        expected = json.loads((GOLDEN_FIXTURES / "expected_rankings.json").read_text(encoding="utf-8"))
        cases = {
            "text": (GOLDEN_FIXTURES / "text" / "input.txt", "text"),
            "csv": (GOLDEN_FIXTURES / "csv" / "input.csv", "csv"),
            "json": (GOLDEN_FIXTURES / "json" / "input.json", "json"),
            "mixed": (GOLDEN_FIXTURES / "mixed", None),
        }

        for name, (input_path, source_type) in cases.items():
            with self.subTest(name=name):
                report = TechniqueEvaluationRunner(config=TechniqueEvaluationConfig()).run(
                    input_path=input_path,
                    source_type=source_type,
                    techniques="applicable",
                )
                ranking = [score.technique_name for score in report.ranked_scores]

                self.assertEqual(ranking, expected[name]["ranking"])

    def test_markdown_report_matches_ranked_json_order_and_score_fields(self) -> None:
        report = TechniqueEvaluationRunner(config=TechniqueEvaluationConfig()).run(
            input_path=GOLDEN_FIXTURES / "csv" / "input.csv",
            source_type="csv",
            techniques="applicable",
        )
        markdown = MarkdownReportWriter().render(report)

        self.assertIn("## Benchmark Metadata", markdown)
        self.assertIn("Quality", markdown)
        self.assertIn("Runtime", markdown)
        self.assertIn(report.run_metadata.config_hash, markdown)
        self.assertLess(
            markdown.index("direct_table_mapper"),
            markdown.index("jsonpath_mapper"),
        )

    def test_benchmark_cli_writes_json_and_markdown_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            markdown_path = Path(tmpdir) / "report.md"
            json_path = Path(tmpdir) / "report.json"
            code = benchmark_main(
                [
                    "--input",
                    "examples/data_to_kg/sample.txt",
                    "--source-type",
                    "text",
                    "--config",
                    "configs/evaluation.example.yaml",
                    "--techniques",
                    "applicable",
                    "--output",
                    str(markdown_path),
                    "--json-output",
                    str(json_path),
                    "--candidate-output-dir",
                    str(Path(tmpdir) / "candidates"),
                ]
            )
            report_payload = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

        self.assertEqual(code, 0)
        self.assertIn("KG Construction Technique Evaluation Report", markdown)
        self.assertIn("## Benchmark Metadata", markdown)
        self.assertIn("Quality", markdown)
        self.assertIn("recommendations", report_payload)
        self.assertIn("run_metadata", report_payload)
        self.assertGreaterEqual(len(report_payload["ranked_scores"]), 3)
        self.assertIn("quality_score", report_payload["ranked_scores"][0])
        self.assertIn("runtime_score", report_payload["ranked_scores"][0])


if __name__ == "__main__":
    unittest.main()
