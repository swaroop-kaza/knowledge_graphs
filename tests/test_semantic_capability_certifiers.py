"""Focused tests for executable per-capability semantic certification."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from kg_construction.certification.capabilities import (
    SCENARIOS,
    CertificationContext,
    _certified_graph_id,
    _composed_canonical_ids,
    _composed_graph_evaluation_suite,
    _composed_source,
    certify_capability,
    uncovered_manifest_capabilities,
)
from kg_construction.certification.execution import validate_execution_report
from kg_construction.certification.preflight import check_execution_plan
from kg_construction.certification.runner import dispatch_capabilities
from kg_construction.domain.graph_objects import (
    PropertyEdge,
    PropertyGraph,
    PropertyNode,
)
from kg_construction.infrastructure.evaluation import (
    GraphBoundIntendedUseEvaluator,
    IntendedUseEvaluationReport,
    IntendedUseMetricsEvaluator,
    KnowledgeGraphQualityEvaluator,
    KnowledgeGraphQualityReport,
    SchemaQualityEvaluator,
    SchemaQualityReport,
    predicted_quality_mappings,
)

_ROOT = Path(__file__).resolve().parents[1]
_LIVE_ONLY = {
    "cross_encoder_resolver",
    "dbpedia_linker",
    "dead_letter_replay_tool",
    "embedding_resolver",
    "evidence_first_unstructured_pipeline",
    "gliner_entity_extractor",
    "gliner_ollama_relation_typing_extractor",
    "gliner_relex_claim_extractor",
    "hybrid_gliner_then_ollama_relations",
    "ladybug_exporter",
    "ladybug_ingestion_adapter",
    "memgraph_exporter",
    "mandatory_claim_ensemble",
    "mandatory_composed_unstructured_runtime",
    "mandatory_mention_ensemble",
    "mlx_qwen_claim_extractor",
    "neo4j_exporter",
    "neo4j_ingestion_adapter",
    "neo4j_vector_store_adapter",
    "ollama_graph_extractor",
    "ollama_provider",
    "ollama_qwen_claim_extractor",
    "openie_relation_extractor",
    "openrefine_reconciler",
    "spacy_ner_extractor",
    "transformer_ner_extractor",
    "transformers_qwen_claim_extractor",
    "wikidata_linker",
}
_MANDATORY_QUALITY_FAMILIES = {
    "formation",
    "structured",
    "category",
    "idempotence",
    "ner",
    "relation",
    "coreference",
    "linking",
    "query",
    "vector",
    "operational",
}
_MANDATORY_INTENDED_USES = {
    "sql_like",
    "nosql",
    "graphrag",
    "vector",
    "link_prediction",
    "recommendation",
    "reasoning",
}


class SemanticScenarioCoverageTests(unittest.TestCase):
    def test_composed_runtime_fixture_executes_all_seven_graph_workloads(self) -> None:
        profile = "darwin-arm64"
        suite = _composed_graph_evaluation_suite(profile)
        self.assertEqual(
            suite.fixture_sha256,
            _composed_graph_evaluation_suite(profile).fixture_sha256,
        )
        self.assertEqual("unstructured", suite.construction_profile)

        source = _composed_source()
        canonical_ids = _composed_canonical_ids()
        external_ids = {
            "Alan Turing": "Q7251",
            "Charles Babbage": "Q46633",
            "Ada Lovelace": "Q7259",
        }
        graph_ids = {
            name: _certified_graph_id(profile, canonical_id)
            for name, canonical_id in canonical_ids.items()
        }
        graph = PropertyGraph(
            nodes=[
                PropertyNode(
                    node_id=graph_ids[name],
                    labels=["Entity", "CanonicalEntity", "Person"],
                    properties={
                        "name": name,
                        "external_id": external_ids[name],
                        "source_ids": [source.source_id],
                    },
                )
                for name in canonical_ids
            ],
            edges=[
                PropertyEdge(
                    edge_id="alan-charles",
                    source_id=graph_ids["Alan Turing"],
                    target_id=graph_ids["Charles Babbage"],
                    type="WORKS_WITH",
                ),
                PropertyEdge(
                    edge_id="charles-ada",
                    source_id=graph_ids["Charles Babbage"],
                    target_id=graph_ids["Ada Lovelace"],
                    type="WORKS_WITH",
                ),
            ],
        )

        report, execution = GraphBoundIntendedUseEvaluator().evaluate(graph, suite)

        self.assertTrue(report.passed)
        self.assertEqual(_MANDATORY_INTENDED_USES, set(report.families))
        self.assertTrue(all(family.passed for family in report.families.values()))
        self.assertEqual(_MANDATORY_INTENDED_USES, set(execution.query_hashes))

        linked_mentions = [
            entity.model_copy(
                update={
                    "properties": {
                        **entity.properties,
                        "linked_canonical_entity_id": canonical_ids[
                            str(entity.properties["surface"])
                        ],
                    }
                }
            )
            for entity in suite.quality.expected_batch.entities
        ]
        predicted_coreference, predicted_links = predicted_quality_mappings(
            suite.quality.expected_batch.model_copy(
                update={"entities": linked_mentions}
            )
        )
        self.assertEqual(
            suite.quality.coreference_clusters,
            predicted_coreference,
        )
        self.assertEqual(suite.quality.entity_links, predicted_links)

    def test_mandatory_evaluators_execute_every_metric_family_with_real_fixtures(
        self,
    ) -> None:
        owners = {
            "knowledge_graph_quality_evaluator": KnowledgeGraphQualityEvaluator,
            "schema_quality_evaluator": SchemaQualityEvaluator,
            "intended_use_metrics_evaluator": IntendedUseMetricsEvaluator,
        }
        with tempfile.TemporaryDirectory() as directory:
            for capability, owner in owners.items():
                with self.subTest(capability=capability):
                    context = CertificationContext(
                        capability=capability,
                        profile="darwin-arm64",
                        run_id=f"unit-{capability}",
                        root=_ROOT,
                        workdir=Path(directory) / capability,
                    )
                    context.workdir.mkdir()
                    result = SCENARIOS[capability](owner, context)

                    self.assertIsInstance(result.instance, owner)
                    self.assertEqual("evaluate", result.operation)
                    self.assertTrue(result.metrics)
                    self.assertEqual(set(result.metrics), set(result.thresholds))
                    self.assertTrue(
                        all(
                            result.metrics[name] >= result.thresholds[name]
                            for name in result.metrics
                        )
                    )
                    self.assertTrue(result.assertions)
                    self.assertTrue(
                        all(assertion.get("passed") is True for assertion in result.assertions)
                    )

                    if capability == "knowledge_graph_quality_evaluator":
                        report = result.output
                        self.assertIsInstance(report, KnowledgeGraphQualityReport)
                        assert isinstance(report, KnowledgeGraphQualityReport)
                        self.assertTrue(report.passed)
                        self.assertEqual([], report.missing_families)
                        self.assertEqual(
                            _MANDATORY_QUALITY_FAMILIES,
                            set(report.families),
                        )
                        self.assertTrue(
                            all(
                                family.executed
                                and family.passed
                                and bool(family.metrics)
                                for family in report.families.values()
                            )
                        )
                    elif capability == "schema_quality_evaluator":
                        report = result.output
                        self.assertIsInstance(report, SchemaQualityReport)
                        assert isinstance(report, SchemaQualityReport)
                        self.assertTrue(report.passed)
                        self.assertTrue(report.ontology_evaluated)
                        self.assertTrue(report.shacl_executed)
                        self.assertEqual(0, report.shacl_violation_count)
                        self.assertEqual(3, report.node_count)
                        self.assertEqual(2, report.edge_count)
                        self.assertEqual(1, report.connected_component_count)
                        self.assertTrue(report.gates)
                        self.assertTrue(all(gate.passed for gate in report.gates))
                    else:
                        report = result.output
                        self.assertIsInstance(report, IntendedUseEvaluationReport)
                        assert isinstance(report, IntendedUseEvaluationReport)
                        self.assertTrue(report.passed)
                        self.assertEqual([], report.missing_uses)
                        self.assertEqual(
                            _MANDATORY_INTENDED_USES,
                            set(report.families),
                        )
                        self.assertTrue(
                            all(
                                family.executed
                                and family.passed
                                and bool(family.metrics)
                                for family in report.families.values()
                            )
                        )

    def test_every_manifest_capability_has_exactly_one_semantic_scenario(self) -> None:
        manifest = yaml.safe_load(
            (_ROOT / "capability_manifest.yaml").read_text(encoding="utf-8")
        )
        declared = {row["name"] for row in manifest["techniques"]}

        self.assertFalse(uncovered_manifest_capabilities(_ROOT))
        self.assertEqual(declared, set(SCENARIOS))

    def test_committed_execution_plan_covers_every_profile_and_capability(self) -> None:
        for profile in ("darwin-arm64", "linux-cpu", "linux-cuda"):
            result = check_execution_plan(
                _ROOT / "capability_manifest.yaml",
                _ROOT / "certification" / "execution_plan.yaml",
                profile=profile,
            )
            self.assertTrue(result.passed, result.detail)

    def test_all_non_live_scenarios_execute_and_pass_semantic_thresholds(self) -> None:
        failures: list[str] = []
        for capability in sorted(set(SCENARIOS) - _LIVE_ONLY):
            with self.subTest(capability=capability):
                try:
                    evidence = certify_capability(
                        capability,
                        profile="darwin-arm64",
                        run_id=f"unit-{capability}",
                        root=_ROOT,
                    )
                except Exception as exc:  # pragma: no cover - collected for useful aggregate output
                    failures.append(f"{capability}: {type(exc).__name__}: {exc}")
                    continue
                observations = evidence["observations"]
                self.assertTrue(observations["instantiated"])
                self.assertTrue(observations["executed"])
                self.assertTrue(observations["evaluated"])
                self.assertTrue(observations["passed"])
                self.assertFalse(observations["skipped"])
                self.assertFalse(observations["fallback"])
                self.assertTrue(observations["evaluated_evidence"]["metrics"])
                self.assertTrue(observations["evaluated_evidence"]["thresholds"])
                self.assertEqual(
                    64,
                    len(observations["executed_evidence"]["fixture_sha256"]),
                )
                self.assertEqual(
                    64,
                    len(observations["executed_evidence"]["output_sha256"]),
                )
        self.assertFalse(failures, "\n".join(failures))

    def test_direct_certifier_rejects_an_inapplicable_profile(self) -> None:
        with self.assertRaisesRegex(ValueError, "not applicable"):
            certify_capability(
                "mlx_qwen_claim_extractor",
                profile="linux-cpu",
                run_id="unit-profile-mismatch",
                root=_ROOT,
            )


class SemanticCertifierProcessTests(unittest.TestCase):
    def test_command_writes_nonce_bound_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence_path = Path(directory) / "evidence.json"
            environment = {
                **os.environ,
                "KG_CERT_CAPABILITY": "semantic_column_profiler",
                "KG_CERT_PROFILE": "darwin-arm64",
                "KG_CERT_RUN_ID": "fresh-test-nonce",
                "KG_CERT_EVIDENCE_PATH": str(evidence_path),
            }
            completed = subprocess.run(
                [sys.executable, "scripts/certify_capability.py"],
                cwd=_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["capability"], "semantic_column_profiler")
            self.assertEqual(evidence["run_id"], "fresh-test-nonce")
            self.assertTrue(evidence["observations"]["passed"])

    def test_dispatcher_report_validates_for_real_semantic_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            manifest = temporary / "manifest.yaml"
            plan = temporary / "plan.yaml"
            report = temporary / "report.yaml"
            names = ["semantic_column_profiler", "regex_entity_extractor"]
            manifest.write_text(
                yaml.safe_dump({"techniques": [{"name": name} for name in names]}),
                encoding="utf-8",
            )
            contract = {
                "profiles": ["darwin-arm64"],
                "command": [sys.executable, "scripts/certify_capability.py"],
                "cwd": str(_ROOT),
            }
            plan.write_text(
                yaml.safe_dump(
                    {
                        "capabilities": {
                            name: dict(contract)
                            for name in names
                        }
                    }
                ),
                encoding="utf-8",
            )

            dispatch_capabilities(
                manifest,
                plan,
                profile="darwin-arm64",
                output_path=report,
            )
            validation = validate_execution_report(
                manifest,
                report,
                expected_profile="darwin-arm64",
            )

            self.assertTrue(validation.passed, validation.detail)

    def test_dispatcher_and_validator_apply_capability_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            manifest = temporary / "manifest.yaml"
            plan = temporary / "plan.yaml"
            manifest.write_text(
                yaml.safe_dump(
                    {
                        "techniques": [
                            {
                                "name": "semantic_column_profiler",
                                "profiles": [
                                    "darwin-arm64",
                                    "linux-cpu",
                                    "linux-cuda",
                                ],
                            },
                            {
                                "name": "regex_entity_extractor",
                                "profiles": ["darwin-arm64"],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            common_contract = {
                "profiles": ["darwin-arm64", "linux-cpu", "linux-cuda"],
                "command": [sys.executable, "scripts/certify_capability.py"],
                "cwd": str(_ROOT),
            }
            darwin_contract = {
                "profiles": ["darwin-arm64"],
                "command": [sys.executable, "scripts/certify_capability.py"],
                "cwd": str(_ROOT),
            }
            plan.write_text(
                yaml.safe_dump(
                    {
                        "capabilities": {
                            "semantic_column_profiler": common_contract,
                            "regex_entity_extractor": darwin_contract,
                        }
                    }
                ),
                encoding="utf-8",
            )

            for profile, expected in (
                ("linux-cpu", {"semantic_column_profiler"}),
                (
                    "darwin-arm64",
                    {"semantic_column_profiler", "regex_entity_extractor"},
                ),
            ):
                report = temporary / f"{profile}.yaml"
                generated = dispatch_capabilities(
                    manifest,
                    plan,
                    profile=profile,
                    output_path=report,
                )
                self.assertEqual(
                    expected,
                    {row["name"] for row in generated["capabilities"]},
                )
                validation = validate_execution_report(
                    manifest,
                    report,
                    expected_profile=profile,
                )
                self.assertTrue(validation.passed, validation.detail)
                self.assertEqual(expected, set(validation.declared))


if __name__ == "__main__":
    unittest.main()
