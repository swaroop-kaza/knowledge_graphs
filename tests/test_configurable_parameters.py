"""Construction profile and evaluation-threshold configuration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from pydantic import ValidationError

from kg_construction import convert
from kg_construction.construction.config import (
    DataToKGConfig,
    EvaluationThresholdsConfig,
    ExtractionConfig,
    QualityThresholdOverride,
)


class ConfigurableExtractionTests(unittest.TestCase):
    def test_extraction_and_threshold_sections_have_defaults(self) -> None:
        config = DataToKGConfig()
        self.assertEqual(config.extraction.runtime_profile, "deterministic")
        self.assertIn("Person", config.extraction.entity_labels)
        self.assertEqual(config.evaluation_thresholds.ner_f1_min, 0.80)
        self.assertEqual(config.evaluation_thresholds.vector_recall_at_k_min, 0.70)

    def test_model_backed_profile_accepts_governed_entity_labels(self) -> None:
        extraction = ExtractionConfig(
            runtime_profile="model_backed",
            entity_labels=("Person", "Gene", "Disease"),
        )
        self.assertEqual(extraction.runtime_profile, "model_backed")
        self.assertEqual(extraction.entity_labels, ("Person", "Gene", "Disease"))

    def test_individual_provider_switches_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ExtractionConfig.model_validate(
                {
                    "runtime_profile": "model_backed",
                    "ner": [{"backend": "gliner"}],
                }
            )

    def test_entity_labels_reject_duplicates(self) -> None:
        with self.assertRaises(ValidationError):
            ExtractionConfig(entity_labels=("Person", "person"))

    def test_config_round_trips_through_yaml(self) -> None:
        payload = {
            "extraction": {
                "runtime_profile": "model_backed",
                "entity_labels": ["Person", "Gene"],
            },
            "evaluation_thresholds": {"ner_f1_min": 0.6},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            config = DataToKGConfig.from_yaml(path)
        self.assertEqual(config.extraction.runtime_profile, "model_backed")
        self.assertEqual(config.extraction.entity_labels, ("Person", "Gene"))
        self.assertEqual(config.evaluation_thresholds.ner_f1_min, 0.6)

    def test_model_backed_failure_does_not_change_runtime_profile(self) -> None:
        config = DataToKGConfig(
            extraction=ExtractionConfig(runtime_profile="model_backed")
        )
        with (
            patch(
                "kg_construction.construction.api.LockedServiceRuntime.from_environment",
                return_value=object(),
            ),
            patch(
                "kg_construction.construction.api._mandatory_unstructured_pipeline",
                side_effect=RuntimeError("provider unavailable"),
            ),
            self.assertRaisesRegex(RuntimeError, "provider unavailable"),
        ):
            convert("Ada founded Acme.", "text", config=config)


class ConfigurableThresholdWiringTests(unittest.TestCase):
    def test_thresholds_are_reported_not_enforced(self) -> None:
        # A stricter-or-looser threshold changes the reported gate but never the
        # fact that a graph is returned.
        config = DataToKGConfig(
            evaluation_thresholds=EvaluationThresholdsConfig(
                ner_f1_min=0.5,
                quality_overrides=(
                    QualityThresholdOverride(
                        family="formation",
                        metric="source_coverage",
                        operator=">=",
                        threshold=0.25,
                    ),
                ),
            )
        )
        result = convert("examples/data_to_kg/sample.csv", "csv", config=config)

        self.assertGreater(len(result.property_graph.nodes), 0)
        self.assertEqual(
            result.quality_report.thresholds_version, "kg-quality-v3+user"
        )
        gates = {
            (family, gate.metric): gate
            for family, report in result.quality_report.families.items()
            for gate in report.gates
        }
        self.assertEqual(gates[("formation", "source_coverage")].threshold, 0.25)
        self.assertEqual(gates[("ner", "entity_f1")].threshold, 0.5)

    def test_default_thresholds_keep_the_default_version(self) -> None:
        result = convert("examples/data_to_kg/sample.csv", "csv")
        self.assertEqual(result.quality_report.thresholds_version, "kg-quality-v3")

    def test_unknown_quality_override_is_rejected(self) -> None:
        config = DataToKGConfig(
            evaluation_thresholds=EvaluationThresholdsConfig(
                quality_overrides=(
                    QualityThresholdOverride(
                        family="formation",
                        metric="not_a_real_metric",
                        operator=">=",
                        threshold=1.0,
                    ),
                )
            )
        )
        with self.assertRaisesRegex(
            ValueError,
            "unknown quality threshold override",
        ):
            convert("examples/data_to_kg/sample.csv", "csv", config=config)


if __name__ == "__main__":
    unittest.main()
