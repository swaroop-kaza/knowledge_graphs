from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.generate_cross_source_identity_benchmark import generate_results

from kg_construction.application.benchmark_runner import (
    BenchmarkHardGateSettings,
    EvaluationSettings,
    TechniqueEvaluationConfig,
    TechniqueEvaluationRunner,
)
from kg_construction.application.scoring import ScoringEngine
from kg_construction.domain.models import (
    ConstructionBatch,
    EntityCandidate,
    RelationCandidate,
)
from kg_construction.infrastructure.evaluation.entity_resolution_metrics import (
    EntityResolutionMetrics,
    predicted_merge_pairs,
)
from kg_construction.infrastructure.evaluation.gold_standards import (
    GoldBenchmarkEvaluator,
    GoldBenchmarkFixture,
    GoldEntityResolutionFixture,
    GoldRetrievalFixture,
)
from kg_construction.infrastructure.reports.csv_report import CsvReportWriter
from kg_construction.infrastructure.reports.json_report import JsonReportWriter
from kg_construction.infrastructure.reports.markdown_report import MarkdownReportWriter


class PublishableBenchmarkTests(unittest.TestCase):
    def test_json_publication_omits_source_derived_artifacts_by_default(self) -> None:
        report = TechniqueEvaluationRunner().run(
            input_path="tests/fixtures/golden_benchmarks/text/input.txt",
            source_type="text",
            techniques="regex_entity_extractor",
        )

        summary = JsonReportWriter().payload(report)
        sensitive = JsonReportWriter(include_sensitive_artifacts=True).payload(report)

        self.assertEqual(summary["publication_serialization"]["mode"], "summary_only")
        for field in ("batch", "property_graph", "rdf_graph", "evidence_graph"):
            self.assertNotIn(field, summary["candidates"][0])
            self.assertIn(field, sensitive["candidates"][0])

    def test_gold_extraction_fixture_changes_accuracy_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            input_path = directory / "input.txt"
            gold_path = directory / "gold.json"
            text = "INCIDENT-1042 affected Recommendation API on 2026-07-09."
            input_path.write_text(text, encoding="utf-8")
            gold_path.write_text(
                json.dumps(
                    {
                        "fixture_id": "unit_gold_extraction",
                        "text": text,
                        "entity_spans": [
                            {"text": "INCIDENT-1042", "label": "Incident", "start": 0, "end": 13},
                            {"text": "Recommendation API", "label": "System", "start": 23, "end": 41},
                            {"text": "2026-07-09", "label": "Date", "start": 45, "end": 55},
                        ],
                        "triples": [],
                        "relations": [],
                        "metadata": {"version": "1.0.0"},
                    }
                ),
                encoding="utf-8",
            )
            config = TechniqueEvaluationConfig(
                evaluation=EvaluationSettings(gold_dataset_path=str(gold_path))
            )
            report = TechniqueEvaluationRunner(config=config).run(
                input_path=input_path,
                source_type="text",
                techniques="hybrid_rules_gazetteer_regex",
            )

        candidate = report.candidates[0]
        # The hybrid finds all three gold spans plus one extra Identifier span.
        # This proves the gold fixture is penalizing a real false positive.
        self.assertEqual(candidate.evaluation_report["entity_f1"], 0.857143)
        self.assertEqual(candidate.evaluation_report["span_f1"], 0.857143)
        self.assertEqual(candidate.evaluation_report["gold_entity_true_positive"], 3.0)
        self.assertEqual(candidate.evaluation_report["gold_entity_false_positive"], 1.0)
        self.assertEqual(
            candidate.gold_evaluation.evaluated_tasks,
            ["entities", "entity_spans", "relations", "triples"],
        )
        self.assertEqual(report.run_metadata.benchmark_id, "unit_gold_extraction")
        self.assertEqual(report.run_metadata.benchmark_version, "1.0.0")
        self.assertIsNotNone(report.run_metadata.gold_dataset_hash)
        self.assertIn("python", report.run_metadata.environment)
        self.assertEqual(len(report.run_metadata.source_tree_hash), 64)
        self.assertIn(report.run_metadata.git_tree_state, {"clean", "dirty", "unknown"})
        self.assertTrue(
            next(gate for gate in candidate.hard_gates if gate.gate == "gold_correctness").passed
        )

    def test_explicit_empty_gold_task_penalizes_false_positives(self) -> None:
        relation = RelationCandidate(
            candidate_id="relation_1",
            source="entity_a",
            relation_type="KNOWS",
            target="entity_b",
            extractor_name="test",
            technique="test",
        )
        declared_negative = GoldBenchmarkFixture(
            fixture_id="reviewed_negative",
            relations=[],
        )
        evaluator = GoldBenchmarkEvaluator()

        false_positive = evaluator.evaluate_relation_mentions(
            ConstructionBatch(relations=[relation]),
            declared_negative,
        )
        correct_negative = evaluator.evaluate_relation_mentions(
            ConstructionBatch(),
            declared_negative,
        )

        self.assertIn("relations", declared_negative.model_fields_set)
        self.assertEqual(false_positive.f1, 0.0)
        self.assertEqual(correct_negative.f1, 1.0)

    def test_span_accuracy_has_a_separate_publication_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            input_path = directory / "input.txt"
            gold_path = directory / "gold.json"
            text = "INCIDENT-1042 affected Recommendation API on 2026-07-09."
            input_path.write_text(text, encoding="utf-8")
            gold_path.write_text(
                json.dumps(
                    {
                        "fixture_id": "span_gate",
                        "text": text,
                        "entity_spans": [
                            {
                                "text": "INCIDENT-1042",
                                "label": "Incident",
                                "start": 0,
                                "end": 13,
                            },
                            {
                                "text": "Recommendation API",
                                "label": "System",
                                "start": 23,
                                "end": 41,
                            },
                            {
                                "text": "2026-07-09",
                                "label": "Date",
                                "start": 45,
                                "end": 55,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config = TechniqueEvaluationConfig(
                evaluation=EvaluationSettings(gold_dataset_path=str(gold_path)),
                hard_gates=BenchmarkHardGateSettings(
                    minimum_entity_f1=0.8,
                    minimum_span_f1=0.9,
                ),
            )
            report = TechniqueEvaluationRunner(config=config).run(
                input_path=input_path,
                source_type="text",
                techniques="hybrid_rules_gazetteer_regex",
            )

        gates = {gate.gate: gate for gate in report.candidates[0].hard_gates}
        self.assertTrue(gates["entity_extraction_accuracy"].passed)
        self.assertFalse(gates["entity_span_accuracy"].passed)
        self.assertFalse(report.candidates[0].publication_eligible)

    def test_gold_entity_resolution_fixture_is_used_by_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            input_path = directory / "input.csv"
            input_path.write_text("id,name\n1,Acme\n2,Acme\n3,Other\n", encoding="utf-8")
            baseline = TechniqueEvaluationRunner().run(
                input_path=input_path,
                source_type="csv",
                techniques="direct_table_mapper",
            )
            baseline_batch = baseline.candidates[0].batch
            assert baseline_batch is not None
            entities = baseline_batch.entities
            gold_path = directory / "er.json"
            gold_path.write_text(
                json.dumps(
                    {
                        "fixture_id": "unit_er",
                        "evaluation_record_id_property": "id",
                        "records": [
                            {
                                "record_id": str(entity.properties["id"]),
                                "entity_type": entity.entity_type,
                                "name": entity.name,
                                "cluster_id": "acme" if entity.name == "Acme" else "other",
                            }
                            for entity in entities
                        ],
                        "metadata": {"version": "1.0.0"},
                    }
                ),
                encoding="utf-8",
            )
            config = TechniqueEvaluationConfig(
                evaluation=EvaluationSettings(gold_dataset_path=str(gold_path))
            )
            report = TechniqueEvaluationRunner(config=config).run(
                input_path=input_path,
                source_type="csv",
                techniques="direct_table_mapper",
            )

        candidate = report.candidates[0]
        # DirectTableMapper emits source-scoped identity keys, so the same-name
        # rows stay split. Gold evaluation must expose that false split.
        self.assertEqual(candidate.evaluation_report["pairwise_er_f1"], 0.0)
        self.assertEqual(candidate.evaluation_report["false_split_rate"], 1.0)
        self.assertEqual(candidate.evaluation_report["er_record_id_overlap_rate"], 1.0)
        self.assertEqual(report.ranked_scores[0].dimensions["accuracy"], 0.0)
        self.assertIn("entity_resolution", candidate.gold_evaluation.evaluated_tasks)
        self.assertTrue(next(gate for gate in candidate.hard_gates if gate.gate == "false_merges").passed)
        self.assertNotIn(
            "No gold labels were supplied",
            " ".join(report.recommendations.limitations),
        )
        self.assertEqual(
            report.recommendations.evaluated_gold_tasks,
            ["entity_resolution"],
        )

    def test_csv_writer_publishes_scores_gates_and_reproducibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.csv"
            report = TechniqueEvaluationRunner().run(
                input_path="tests/fixtures/golden_benchmarks/text/input.txt",
                source_type="text",
                techniques="regex_entity_extractor",
            )
            CsvReportWriter().write(report, output_path)
            with output_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        regex_row = next(row for row in rows if row["technique"] == "regex_entity_extractor")
        self.assertIn("gold_correctness", regex_row["hard_gates_failed"])
        self.assertEqual(regex_row["input_hash"], report.run_metadata.input_hash)
        self.assertEqual(regex_row["config_hash"], report.run_metadata.config_hash)

    def test_missing_retrieval_predictions_are_scored_as_zero_and_gated(self) -> None:
        config = TechniqueEvaluationConfig(
            evaluation=EvaluationSettings(
                gold_dataset_path="tests/fixtures/gold_standards/retrieval_cases.json"
            )
        )
        report = TechniqueEvaluationRunner(config=config).run(
            input_path="tests/fixtures/golden_benchmarks/text/input.txt",
            source_type="text",
            techniques="regex_entity_extractor",
        )

        for candidate in report.candidates:
            self.assertEqual(candidate.gold_evaluation.evaluated_tasks, ["retrieval"])
            self.assertEqual(candidate.gold_evaluation.unavailable_tasks, [])
            self.assertEqual(candidate.evaluation_report["retrieval_recall_at_k"], 0.0)
            self.assertEqual(
                candidate.evaluation_report["retrieval_prediction_completeness"],
                0.0,
            )
            self.assertEqual(candidate.gold_evaluation.support["retrieval_predictions"], 0)
            self.assertEqual(
                candidate.gold_evaluation.support["retrieval_missing_predictions"],
                3,
            )
            gold_gate = next(
                gate for gate in candidate.hard_gates if gate.gate == "gold_correctness"
            )
            completeness_gate = next(
                gate
                for gate in candidate.hard_gates
                if gate.gate == "retrieval_prediction_completeness"
            )
            self.assertTrue(gold_gate.passed)
            self.assertFalse(completeness_gate.passed)
            self.assertFalse(candidate.publication_eligible)
        self.assertEqual(report.recommendations.evaluated_gold_tasks, ["retrieval"])
        self.assertIsNone(
            report.recommendations.production_safe_technique.technique_name
        )

    def test_partial_retrieval_predictions_include_missing_queries_as_zero(self) -> None:
        gold = GoldRetrievalFixture.from_json_file(
            "tests/fixtures/gold_standards/retrieval_cases.json"
        )
        batch = ConstructionBatch(
            metadata={
                "retrieval_predictions": {
                    "q_affected_system": {
                        "retrieved_context_ids": ["ctx_incident_affected"],
                        "path": [
                            "entity_incident_1042",
                            "AFFECTED",
                            "entity_recommendation_api",
                        ],
                        "citation_ids": ["evidence_incident_affected"],
                        "answer": "Recommendation API",
                    }
                }
            }
        )

        metrics = TechniqueEvaluationRunner._evaluate_gold_retrieval(
            SimpleNamespace(batch=batch),
            gold,
        )

        self.assertEqual(metrics["retrieval_gold_query_count"], 3.0)
        self.assertEqual(metrics["retrieval_predicted_query_count"], 1.0)
        self.assertEqual(metrics["retrieval_missing_query_count"], 2.0)
        self.assertAlmostEqual(metrics["retrieval_prediction_completeness"], 1 / 3)
        self.assertAlmostEqual(metrics["retrieval_recall_at_k"], 1 / 3)

    def test_disjoint_er_record_id_universes_fail_loudly(self) -> None:
        batch = ConstructionBatch(
            entities=[
                EntityCandidate(
                    candidate_id="internal_1",
                    entity_type="Organization",
                    name="Acme",
                    properties={"record_id": "source_1"},
                    extractor_name="test",
                    technique="test",
                )
            ]
        )

        with self.assertRaisesRegex(ValueError, "record-ID universes are disjoint"):
            EntityResolutionMetrics().evaluate(
                batch,
                gold_pairs=set(),
                gold_record_ids={"human_1"},
                record_id_property="record_id",
            )

    def test_partial_er_record_id_coverage_fails_publication_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            input_path = directory / "input.csv"
            gold_path = directory / "er.json"
            input_path.write_text("id,name\n1,Acme\n2,Acme\n3,Other\n", encoding="utf-8")
            gold_path.write_text(
                json.dumps(
                    {
                        "fixture_id": "partial_universe",
                        "evaluation_record_id_property": "id",
                        "records": [
                            {
                                "record_id": "1",
                                "entity_type": "Record",
                                "name": "Acme",
                                "cluster_id": "acme",
                            },
                            {
                                "record_id": "2",
                                "entity_type": "Record",
                                "name": "Acme",
                                "cluster_id": "acme",
                            },
                            {
                                "record_id": "4",
                                "entity_type": "Record",
                                "name": "Missing",
                                "cluster_id": "missing",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config = TechniqueEvaluationConfig(
                evaluation=EvaluationSettings(gold_dataset_path=str(gold_path))
            )
            report = TechniqueEvaluationRunner(config=config).run(
                input_path=input_path,
                source_type="csv",
                techniques="direct_table_mapper",
            )

        candidate = report.candidates[0]
        self.assertAlmostEqual(
            candidate.evaluation_report["er_record_id_overlap_rate"],
            2 / 3,
        )
        coverage_gate = next(
            gate
            for gate in candidate.hard_gates
            if gate.gate == "entity_resolution_record_id_coverage"
        )
        self.assertFalse(coverage_gate.passed)
        self.assertFalse(candidate.publication_eligible)

    def test_manifest_source_inventory_binds_cross_source_runner_input(self) -> None:
        manifest_path = Path(
            "benchmarks/cross_source_identity/v1/benchmark_manifest.yaml"
        )
        config = TechniqueEvaluationConfig(
            evaluation=EvaluationSettings(gold_dataset_path=str(manifest_path))
        )

        report = TechniqueEvaluationRunner(config=config).run(
            input_path=manifest_path.parent,
            techniques="direct_table_mapper",
        )

        candidate = report.candidates[0]
        assert candidate.batch is not None
        self.assertEqual(len(candidate.batch.source_records), 3)
        self.assertEqual(len(candidate.batch.entities), 12)
        self.assertEqual(
            candidate.evaluation_report["er_record_id_symmetric_coverage"],
            1.0,
        )
        self.assertEqual(
            report.run_metadata.input_hash,
            "36acf7cb9eedd83454f647f1542c8990474dd2231bf5212be40e0db21689fb0b",
        )
        self.assertTrue(report.run_metadata.input_contract_verified)
        self.assertEqual(
            sorted(report.run_metadata.source_inventory),
            ["billing.csv", "crm.csv", "support.csv"],
        )

    def test_manifest_input_hash_rejects_modified_source_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            (directory / "input.csv").write_text(
                "record_id,name\nrecord_1,Acme\n",
                encoding="utf-8",
            )
            (directory / "gold.json").write_text(
                json.dumps(
                    {
                        "fixture_id": "bound_input",
                        "evaluation_record_id_property": "record_id",
                        "records": [
                            {
                                "record_id": "record_1",
                                "entity_type": "Record",
                                "name": "Acme",
                                "cluster_id": "acme",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest_path = directory / "benchmark_manifest.yaml"
            manifest_path.write_text(
                "\n".join(
                    [
                        'benchmark_id: "bound_input"',
                        'version: "1.0.0"',
                        f'input_hash: "{"0" * 64}"',
                        "source_inventory:",
                        "  - input.csv",
                        "entity_resolution_fixture: gold.json",
                    ]
                ),
                encoding="utf-8",
            )
            config = TechniqueEvaluationConfig(
                evaluation=EvaluationSettings(
                    gold_dataset_path=str(manifest_path)
                )
            )

            with self.assertRaisesRegex(ValueError, "input hash does not match"):
                TechniqueEvaluationRunner(config=config).run(
                    input_path=directory,
                    techniques="direct_table_mapper",
                )

    def test_duplicate_candidate_occurrences_do_not_create_self_pairs(self) -> None:
        entity = EntityCandidate(
            candidate_id="record_1",
            entity_type="Person",
            name="Jordan Lee",
            properties={"canonical_entity_id": "person_jordan"},
            extractor_name="test",
            technique="test",
        )
        batch = ConstructionBatch(entities=[entity, entity])

        self.assertEqual(predicted_merge_pairs(batch), set())

    def test_empty_er_gold_fixture_is_rejected(self) -> None:
        fixture = GoldEntityResolutionFixture(
            fixture_id="empty",
            records=[],
        )

        with self.assertRaisesRegex(ValueError, "at least one record"):
            fixture.record_ids()

    def test_gold_zero_accuracy_is_not_hidden_by_estimated_precision(self) -> None:
        score = ScoringEngine().score(
            technique_name="failed_gold_extraction",
            pipeline_name="unit",
            metrics={
                "gold_entities_evaluated": 1.0,
                "entity_f1": 0.0,
                "span_f1": 0.0,
                "estimated_precision": 1.0,
            },
        )

        self.assertEqual(score.dimensions["accuracy"], 0.0)
        self.assertEqual(score.confidence_interval_method, "not_estimated")
        self.assertEqual(
            score.confidence_interval,
            (score.final_score, score.final_score),
        )

    def test_no_gold_override_remains_smoke_only_and_not_production_safe(self) -> None:
        config = TechniqueEvaluationConfig(
            hard_gates=BenchmarkHardGateSettings(require_gold_dataset=False)
        )
        report = TechniqueEvaluationRunner(config=config).run(
            input_path="tests/fixtures/golden_benchmarks/text/input.txt",
            source_type="text",
            techniques="regex_entity_extractor",
        )

        candidate = report.candidates[0]
        self.assertTrue(all(gate.passed for gate in candidate.hard_gates))
        self.assertFalse(candidate.publication_eligible)
        self.assertIsNotNone(report.recommendations.best_single_technique.technique_name)
        self.assertIsNone(
            report.recommendations.production_safe_technique.technique_name
        )

    def test_markdown_does_not_label_proxy_counts_as_score_confidence(self) -> None:
        report = TechniqueEvaluationRunner().run(
            input_path="tests/fixtures/golden_benchmarks/text/input.txt",
            source_type="text",
            techniques="regex_entity_extractor",
        )

        markdown = MarkdownReportWriter().render(report)

        self.assertIn("statistical score uncertainty was not estimated", markdown)
        self.assertNotIn("Wilson", markdown)

    def test_run_metadata_redacts_secrets_and_absolute_external_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir).resolve() / "private-input.txt"
            input_path.write_text("INCIDENT-1042", encoding="utf-8")
            config = TechniqueEvaluationConfig(
                llm={
                    "provider": "remote",
                    "api_key": "must-not-leak",
                    "max_tokens": 128,
                    "nested": {"access-token": "also-secret", "model": "test"},
                }
            )
            report = TechniqueEvaluationRunner(config=config).run(
                input_path=input_path,
                source_type="text",
                techniques="regex_entity_extractor",
            )

        llm_metadata = report.run_metadata.model_settings["llm"]
        self.assertEqual(llm_metadata["api_key"], "[REDACTED]")
        self.assertEqual(llm_metadata["nested"]["access-token"], "[REDACTED]")
        self.assertEqual(llm_metadata["nested"]["model"], "test")
        self.assertEqual(llm_metadata["max_tokens"], 128)
        self.assertEqual(report.input_path, "<external>/private-input.txt")

    def test_cross_source_identity_publication_is_honest_and_reproducible(self) -> None:
        first_rows, first_metadata = generate_results()
        second_rows, second_metadata = generate_results()

        self.assertEqual(first_rows, second_rows)
        self.assertEqual(first_metadata["dataset_hash"], second_metadata["dataset_hash"])
        by_name = {row["baseline"]: row for row in first_rows}
        self.assertEqual(by_name["exact_normalized_type_and_name"]["pairwise_f1"], 0.333333)
        self.assertFalse(by_name["curated_alias_table"]["passes_all_gates"])
        self.assertEqual(by_name["curated_alias_table"]["false_merge_rate"], 0.2)
        self.assertTrue(by_name["trusted_identifier_exact"]["passes_all_gates"])
        self.assertEqual(by_name["trusted_identifier_exact"]["evidence_coverage"], 1.0)
        self.assertEqual(first_metadata["gold_support"]["positive_pairs"], 8)
        self.assertEqual(
            first_metadata["input_hash"],
            "36acf7cb9eedd83454f647f1542c8990474dd2231bf5212be40e0db21689fb0b",
        )
        self.assertEqual(len(first_metadata["source_tree_hash"]), 64)
        self.assertEqual(len(first_metadata["results_hash"]), 64)
        self.assertIn("pydantic", first_metadata["environment"])

        committed = json.loads(
            Path("reports/cross_source_identity_benchmark.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(committed["results"], first_rows)
        for field in (
            "dataset_hash",
            "input_hash",
            "manifest_hash",
            "gold_hash",
            "config_hash",
            "generator_hash",
            "source_tree_hash",
            "results_hash",
        ):
            self.assertEqual(committed["metadata"][field], first_metadata[field])
        self.assertEqual(
            committed["metadata"]["source_hashes"],
            first_metadata["source_hashes"],
        )

        with Path("reports/cross_source_identity_benchmark.csv").open(
            encoding="utf-8",
            newline="",
        ) as handle:
            committed_csv = list(csv.DictReader(handle))
        expected_csv = [
            {key: str(value) for key, value in row.items()}
            for row in first_rows
        ]
        self.assertEqual(committed_csv, expected_csv)

        committed_markdown = Path(
            "reports/cross_source_identity_benchmark.md"
        ).read_text(encoding="utf-8")
        for row in first_rows:
            self.assertIn(
                f"| `{row['baseline']}` | {row['predicted_pair_count']} | ",
                committed_markdown,
            )
        self.assertIn(first_metadata["results_hash"], committed_markdown)


if __name__ == "__main__":
    unittest.main()
