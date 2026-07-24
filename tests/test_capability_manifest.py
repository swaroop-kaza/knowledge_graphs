"""The certified manifest contains only mandatory executable capabilities."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import yaml

from kg_construction.application.registry import default_registry

_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST = _ROOT / "capability_manifest.yaml"
_RESEARCH_INVENTORY = _ROOT / "certification" / "research_inventory.yaml"
_GENERATOR = _ROOT / "scripts" / "generate_capability_manifest.py"
_RENAMES = {
    "kuzu_exporter": "ladybug_exporter",
    "kuzu_ingestion_adapter": "ladybug_ingestion_adapter",
}


def _load_manifest() -> dict:
    return yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))


def _load_generator():
    spec = importlib.util.spec_from_file_location("capability_manifest_generator", _GENERATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ManifestCompletenessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = _load_manifest()
        self.registry = default_registry()
        self.techniques = {row["name"]: row for row in self.manifest["techniques"]}

    def test_manifest_covers_exactly_the_executable_registry(self) -> None:
        executable = {
            _RENAMES.get(row.name, row.name)
            for row in self.registry.techniques
            if row.status != "stub"
        }
        self.assertEqual(set(self.techniques), executable)

    def test_every_disposition_is_mandatory(self) -> None:
        for row in self.manifest["techniques"]:
            self.assertEqual(row["disposition"], "mandatory", row["name"])

    def test_no_stub_planned_or_optional_status_enters_certification(self) -> None:
        forbidden = {"stub", "planned", "optional"}
        for row in self.manifest["techniques"]:
            self.assertNotIn(str(row["status"]).lower(), forbidden, row["name"])

    def test_every_mandatory_capability_has_an_owner(self) -> None:
        for row in self.manifest["techniques"]:
            self.assertTrue(row["owner"], f"mandatory capability {row['name']} has no owner")

    def test_research_inventory_is_disjoint_from_the_release(self) -> None:
        inventory = yaml.safe_load(_RESEARCH_INVENTORY.read_text(encoding="utf-8"))
        research = {row["name"] for row in inventory["techniques"]}
        self.assertFalse(research & set(self.techniques))
        self.assertTrue(all(row["certified"] is False for row in inventory["techniques"]))
        broad = {
            _RENAMES.get(row.name, row.name)
            for row in _load_generator().research_registry().techniques
        }
        self.assertEqual(set(self.techniques) | research, broad)

    def test_summary_counts_match_rows(self) -> None:
        for section, summary in self.manifest["summary"].items():
            rows = self.manifest[section]
            mandatory = sum(1 for row in rows if row["disposition"] == "mandatory")
            self.assertEqual(summary["total"], len(rows), section)
            self.assertEqual(summary["mandatory"], mandatory, section)
            self.assertEqual(summary["total"], summary["mandatory"], section)

    def test_committed_manifest_matches_regeneration(self) -> None:
        generator = _load_generator()
        fresh = generator.build_manifest()
        committed = {key: value for key, value in self.manifest.items()}
        self.assertEqual(
            fresh,
            committed,
            "capability_manifest.yaml is stale; run scripts/generate_capability_manifest.py",
        )
        research = yaml.safe_load(_RESEARCH_INVENTORY.read_text(encoding="utf-8"))
        self.assertEqual(generator.build_research_inventory(), research)


if __name__ == "__main__":
    unittest.main()
